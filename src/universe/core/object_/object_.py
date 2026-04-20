import hjson as json  # type: ignore
from typing import TYPE_CHECKING, Type, Generic, TypeVar, cast
from pydantic import BaseModel, Field

from ..meta.generics_ import GenericsMeta
from ..timing import Timing, TimedStr
from ..llm_client import estimate_tokens
from .serializable import Serializable


O = TypeVar("O", bound="Object")
P = TypeVar("P", bound="Params")


if TYPE_CHECKING:
    from ..universe import World
    from ..agent import Agent


class Params(BaseModel):
    pass


class Action(Generic[O, P], metaclass=GenericsMeta):

    name: str
    description: str

    @classmethod
    def GetParamsType(cls) -> Type[P]:
        return cls._generics[1]
    
    @classmethod
    def GetParams(cls, arguments: dict) -> P:
        return cls.GetParamsType().model_validate_json(**arguments)

    def get_llm_tool_definition(self) -> dict:
        """获取 OpenAI function calling 格式的工具定义"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.GetParamsType().model_json_schema(),
            },
        }

    def execute(self, obj: O, params: P, world: World) -> TimedStr:
        raise NotImplementedError("Subclasses of Action must implement execute()")


class ActionExecutionPackage(BaseModel):
    """来自同一执行者和目标者的动作请求包"""
    actor_id: str              # 动作发送者 ID
    channel: Channel           # 动作通道定义
    tool_calls: list[dict]     # 实际要执行的 tool_call 列表
    tool_call_results: dict[str, str] = Field(default_factory=dict) # 工具调用结果列表 tool_call_id -> result
    resources_ready_time: int | None = None  # 资源就绪时间，None 表示尚未就绪
    resources_efficiency: float | None = Field(default=None, gt=0, le=1)  # 资源效率（0-1之间），None 表示尚未就绪
    action_duration: int | None = None  # 动作执行时间， None 表示尚未执行完成

    def get_resources_ready_time(self) -> int:
        assert self.resources_ready_time is not None, "Resources ready time is not set"
        return self.resources_ready_time
    
    def get_resources_efficiency(self) -> float:
        assert self.resources_efficiency is not None, "Resources efficiency is not set"
        return self.resources_efficiency
    
    def get_action_duration(self) -> int:
        assert self.action_duration is not None, "Action duration is not set"
        return self.action_duration


class Channel(BaseModel):
    cognitive_target: str       # 认知的目标对象别名
    target_id: str              # 动作接收者 ID
    budget: int | None = None   # 观察上下文预算，默认无限预算
    allowed_actions: list[str] | None = None  # 允许的工具调用名称列表，默认允许所有工具

    def get_action(self, action_name: str, world: Object) -> Action:
        """获取对象的指定动作"""
        assert self.allowed_actions is None or action_name in self.allowed_actions, f"Action name {action_name} not allowed in channel {self.allowed_actions}"
        target = world.objects[self.target_id]
        action = target.actions[action_name]
        return action
    
    def has_action(self, action_name: str, world: Object) -> bool:
        return action_name in self.get_allowed_actions(world)
    
    def get_allowed_actions(self, world: Object) -> list[str]:
        if self.allowed_actions:
            return self.allowed_actions
        target = world.objects[self.target_id]
        return list(target.actions.keys())


class Object(Serializable):
    """对象基类"""
    
    DEFAULT_READ_SPEED: float = 10

    object_id: str
    actions: dict[str, Action]
    read_speed: float

    def __init__(self, object_id: str, *,
                 actions: list[Action] | None = None,
                 read_speed: float | None = None,
                ):
        super().__init__()
        self.object_id = object_id
        self.actions = {action.name: action for action in (actions or [])}
        self.read_speed = read_speed or self.DEFAULT_READ_SPEED
        
    @property
    def objects(self) -> dict[str, Object]:
        return cast(dict[str, Object], self._objects)
    
    def _observe_duration(self, content: str, world: World | None = None, observer_id: str | None = None) -> int:
        """计算观察对象状态的持续时间"""
        token_count = estimate_tokens(content)
        read_speed_gain: float
        if world is None or observer_id is None:
            read_speed_gain = 1.0
        else:
            observer = world.objects[observer_id]
            if isinstance(observer, Agent):
                read_speed_gain = observer.read_speed_gain
            else:
                read_speed_gain = 1.0
        return int(token_count / (self.read_speed * read_speed_gain))
    
    def _validate_action_packages(self, packages: list[ActionExecutionPackage]):
        """验证动作请求包满足约束"""

        # 检查调用工具名称是否在通道允许列表中
        for package in packages:
            if package.channel.allowed_actions is None:
                continue
            for tool_call in package.tool_calls:
                action_name = tool_call["function"]["name"]
                if action_name not in package.channel.allowed_actions:
                    raise ValueError(f"Action {action_name} not allowed in channel {package.channel.allowed_actions}")

        # 检查 target_id 与 当前对象 object_id 是否一致
        for package in packages:
            if package.channel.target_id != self.object_id:
                raise ValueError(f"Target ID {package.channel.target_id} does not match object ID {self.object_id}")
            if package.resources_ready_time is None:
                raise ValueError(f"Resource ready time for Target ID {package.channel.target_id} is not set")
            if package.resources_efficiency is None:
                raise ValueError(f"Resource efficiency for Target ID {package.channel.target_id} is not set")

    async def _execute_action(self, tool_call: dict, world: World) -> TimedStr:
        """执行工具调用"""
        action_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        action = self.actions.get(action_name)
        if action is None:
            raise ValueError(f"Action {action_name} not found in object.actions")
        params = action.GetParams(arguments)
        result = action.execute(self, params, world)
        return result

    async def active(self, world: World) -> list[ActionExecutionPackage]:
        """主动阶段（默认实现为空，子类可重写）
         
        职责一：模拟状态随时间的自然演化（更新自身状态到当前世界时间，无需返回动作请求）
            例如：
            - 无控制状态转移
            - 惯性运动更新位置
            - 随着年龄衰老
            - 资源自然消耗
            - 过期数据清理
            
        职责二：通过返回值向其他对象发送动作请求（参考 Agent 类的实现）
        """
        return []

    async def passive(self, packages: list[ActionExecutionPackage], world: World) -> list[ActionExecutionPackage]:
        """被动阶段 - 接收外部动作影响
        """
        packages = await self.arbitrate(packages, world)

        self._validate_action_packages(packages)
        
        packages.sort(key=lambda x: x.get_resources_ready_time())  # 按资源就绪时间排序

        for package in packages:
            action_duration = 0
            for tool_call in package.tool_calls:
                action_result = await self._execute_action(tool_call, world)
                action_duration += action_result.duration
                package.tool_call_results[tool_call["id"]] = action_result.content or ""
            package.action_duration = int(action_duration / package.get_resources_efficiency())
        return packages

    async def observe(self, *, channel: Channel | None = None, world: World | None = None, observer_id: str | None = None) -> TimedStr:
        """观察对象状态，将被嵌入到 LLM 的上下文信息中（感知马尔可夫毯可在此实现）"""
        content = json.dumps(self.state_dict(), ensure_ascii=False)
        duration = self._observe_duration(content, world, observer_id)
        return TimedStr(duration=duration, content=content)

    async def arbitrate(self, packages: list[ActionExecutionPackage], world: World) -> list[ActionExecutionPackage]:
        """仲裁阶段 - 处理同时发起的多个动作请求的冲突或叠加（默认透传+资源立即就绪+100%空闲，子类可重写）（效应马尔可夫毯可在此实现）
        """
        for package in packages:
            if package.resources_ready_time is None:
                package.resources_ready_time = world.time
                package.resources_efficiency = 1.0
        return packages