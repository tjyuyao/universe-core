import warnings
import json
from uuid import uuid4
from typing import TYPE_CHECKING, Any

from ..object_ import Object, Action, Channel, Activity, State, ActionExecutionContext, Params, ActionExecutionStatus, TimedStatus
from ..llm_client import LLMClient, LLMResult, estimate_tokens, BudgetWarning, ToolCall
from ..timing import TimedStr

from .soul import Soul
from .role import Role
from .mindset import Mindset
from .attention import Attention


if TYPE_CHECKING:
    from ..universe import World


_llm_client = LLMClient()


class Agent(Object):

    DEFAULT_READ_SPEED_GAIN: float = 1.0
    DEFAULT_THINK_SPEED_GAIN: float = 1.0

    read_speed_gain: State[float]
    think_speed_gain: State[float]
    attention: State[Attention]

    def __init__(self, agent_id: str, *,
                 actions: list[Action] | None = None,
                 read_speed: float | None = None,  # as object read by others (tokens per second)
                 read_speed_gain: float | None = None,  # active reading speed gain
                 think_speed_gain: float | None = None,  # active thinking speed gain
                ):
        super().__init__(
            object_id=agent_id,
            actions=actions,
            read_speed=read_speed,
        )
        self.read_speed_gain = read_speed_gain or self.DEFAULT_READ_SPEED_GAIN
        self.think_speed_gain = think_speed_gain or self.DEFAULT_THINK_SPEED_GAIN

        self.attention = Attention()

    @property
    def agent_id(self):
        return self.object_id

    def append_busy_time(self, duration: float) -> None:
        self._busy_until += duration

    def _build_system_prompt(self, world: World) -> str:
        """构建系统提示"""
        return f"""用户消息中包含最新的完整上下文。你需要根据上下文，深度扮演符合接下来描述的角色的人格，具体是指你需要做出符合角色性格、能力、情绪、思维模式的工具调用来与世界交互。只有工具调用是必要的输出，输出的 content 没有任何额外效用；但对于困难的问题，在 content 中包含临时的中间思考过程是适宜的；但长思考会影响动作发出的时延，因此如果上下文情境处于紧急场合，则不应过度思考。角色的描述如下：
Your Identity: {self.agent_id}
World({world.name}): {world.description}
Soul({self.attention.get_current_soul().name}): {self.attention.get_current_soul().description}
Role({self.attention.get_current_role().name}): {self.attention.get_current_role().description}
Mindset({self.attention.get_current_mindset().name}): {self.attention.get_current_mindset().description}
"""

    async def _build_user_prompt(self, world: World) -> TimedStr:
        """构建用户提示"""
        observe_duration = 0.0
        contexts = {}
        channels = self.attention.get_current_channels()
        model_name = self.attention.get_current_model_name()
        for channel in channels.values():
            target = world.objects[channel.target_id]
            assert isinstance(target, Object)
            context = await target.observe(channel=channel, world=world, observer_id=self.agent_id)
            observe_duration += context.duration  # observe duration
            if channel.budget is not None:
                token_count = estimate_tokens(context, model=model_name)
                if token_count > channel.budget:
                    warnings.warn(BudgetWarning(token_count, channel.budget, channel.cognitive_target))
                contexts[channel.cognitive_target] = context
        return TimedStr(
            duration=observe_duration,
            content=f"""你当前能意识和观察到的完整上下文信息如下，请阅读后决定工具调用行为：
{contexts}""")

    def _build_tools(self, world: World) -> list[dict]:
        tools = []

        # 按 action 名称分组
        action_groups: dict[str, list[Channel]] = {}
        for channel in self.attention.get_current_channels().values():
            if channel.allowed_actions is None:
                continue
            for action_name in channel.allowed_actions:
                action_groups.setdefault(action_name, []).append(channel)

        # 检查 action 名称是否重复
        for action_name, channels in action_groups.items():
            classes = set(type(ch.get_action(action_name, world)) for ch in channels)
            assert len(classes) == 1, (
                f"Action name conflict: '{action_name}' in multiple classes: "
                f"{[cls.__name__ for cls in classes]}"
            )

        # 生成工具定义
        for action_name, channels in action_groups.items():
            channel = channels[0]
            action = channel.get_action(action_name, world)
            base_tool = action.get_llm_tool_definition(channel, world)

            if len(channels) > 1:
                target_names = [ch.cognitive_target for ch in channels]
                tool_def = self._inject_target_param(base_tool, target_names)
            else:
                tool_def = base_tool

            tools.append(tool_def)

        return tools

    def _inject_target_param(self, tool_def: dict, target_names: list[str]) -> dict:
        """向 tool 定义注入 target 参数"""
        import copy
        new_def = copy.deepcopy(tool_def)
        params = new_def["function"]["parameters"]

        params["properties"]["target"] = {
            "type": "string",
            "description": f"动作目标，可选: {target_names}",
            "enum": target_names
        }

        if "required" not in params:
            params["required"] = []
        if "target" not in params["required"]:
            params["required"].insert(0, "target")

        return new_def

    def _parse_response(self, response: LLMResult, world: World) -> dict[str, list[ToolCall]]:
        """解析 LLM 响应，提取工具调用"""

        cog_tar_tool_calls: dict[str, list[ToolCall]] = {}  # cognitive_target -> list[tool_call]
        all_channels = self.attention.get_current_channels()

        for tool_call in response.tool_calls:
            arguments:dict|str = tool_call["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            assert isinstance(arguments, dict)

            action_name = tool_call["name"]

            # 查找支持该 action 的所有 Channel
            supporting_channels = [
                ch for ch in all_channels.values()
                if ch.has_action(action_name, world)
            ]

            cognitive_target = arguments.get("target")

            if cognitive_target is None:
                if len(supporting_channels) == 1:
                    cognitive_target = supporting_channels[0].cognitive_target
                else:
                    available = [ch.cognitive_target for ch in supporting_channels]
                    raise ValueError(
                        f"Missing 'target' for tool: {action_name}. "
                        f"Available: {available}"
                    )

            cog_tar_tool_calls.setdefault(cognitive_target, []).append(tool_call)

        return cog_tar_tool_calls

    async def react(
        self,
        world: World,
        ) -> None:
        """推理和决策阶段（LLM Call）
        """

        # 0. 工具集与Agent自身状态转移
        for channel in self.attention.get_current_channels().values():
            target = world.objects[channel.target_id]
            assert isinstance(target, Object)
            await target.transit(world)
            self._busy_until = max(self._busy_until, target._busy_until)

        if self.is_busy_at(world.time):
            return

        await self.transit(world)

        # 1. 构建 LLM 调用参数 (Observe)
        model_name = self.attention.get_current_model_name()
        system_prompt = self._build_system_prompt(world)
        user_prompt = await self._build_user_prompt(world)  #  call observe()
        tools = self._build_tools(world)
        self.append_busy_time(user_prompt.duration)

        # 2. 调用 LLM (Think)
        response: LLMResult = await _llm_client.complete(
            model=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt.content,
            tools=tools,
            think_speed_gain=self.think_speed_gain,
        )
        self.append_busy_time(response.duration)

        # 3. 解析 LLM 响应 (Act)
        cog_tar_tool_calls = self._parse_response(response, world)
        all_channels = self.attention.get_current_channels()
        for cognitive_target, tool_calls in cog_tar_tool_calls.items():
            action_contexts = {
                str(uuid4()) : ActionExecutionContext(tool_call=tc) for tc in tool_calls
            }
            activity = Activity(
                actor_id=self.agent_id,
                channel=all_channels[cognitive_target],
                action_invoke_time=self._busy_until,
                action_contexts=action_contexts,
            )
            target = world.objects[activity.channel.target_id]
            await target.act(activity)

    def add_soul(self, soul: Soul):
        self.attention.add_soul(soul)

    def get_soul(self, name: str):
        return self.attention.get_soul(name)

    def remove_soul(self, name: str) -> Soul:
        return self.attention.remove_soul(name)

    def add_role(self, soul_name:str, role: Role):
        self.attention.get_soul(soul_name).add_role(role)

    def get_role(self, soul_name:str, role_name:str):
        return self.attention.get_soul(soul_name).get_role(role_name)

    def remove_role(self, soul_name:str, role_name:str) -> Role:
        return self.attention.get_soul(soul_name).remove_role(role_name)

    def add_mindset(self, soul_name:str, role_name:str, mindset: Mindset):
        self.attention.get_soul(soul_name).get_role(role_name).add_mindset(mindset)

    def get_mindset(self, soul_name:str, role_name:str, name: str):
        return self.attention.get_soul(soul_name).get_role(role_name).get_mindset(name)

    def remove_mindset(self, soul_name:str, role_name:str, name: str) -> Mindset:
        return self.attention.get_soul(soul_name).get_role(role_name).remove_mindset(name)
    

class SwitchMindsetToParams(Params):
    mindset_name: str

    @classmethod
    def param_json_schema(cls, channel: Channel, world: World) -> dict[str, Any]:
        schema = super().param_json_schema(channel, world)
        agent = world.objects[channel.target_id]
        assert isinstance(agent, Agent)
        available_mindsets = list(agent.attention.get_current_role().mindsets.keys())
        schema["properties"]["mindset_name"]["enum"] = available_mindsets
        return schema


class SwitchMindsetToAction(Action[Agent, SwitchMindsetToParams]):
    name = "switch_mindset_to"
    description = "切换到其它思维模式对应的状态和行为空间，必须作为当前轮的最后一次工具调用，其后所有工具调用将被忽略"
    
    async def execute(self, obj, params, actor, world):
        assert isinstance(obj, Agent)
        try:
            obj.attention.get_current_role().get_mindset(params.mindset_name)
        except KeyError:
            return TimedStatus(duration=0.1, status=ActionExecutionStatus.FAIL, terminal=True)
        obj.attention.current_mindset = params.mindset_name
        return TimedStatus(duration=5.0, status=ActionExecutionStatus.SUCCESS, terminal=True)
        
