import hjson as json  # type: ignore
from typing import TYPE_CHECKING, Type, Generic, TypeVar, cast
from pydantic import BaseModel, Field

from ..meta.generics_ import GenericsMeta
from ..timing import TimedStr
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
    
    def execute(self, obj: O, params: P, actor: Agent, world: World) -> TimedStr:
        raise NotImplementedError("Subclasses of Action must implement execute() or arbitrate()")


class ActionExecutionPackage(BaseModel):
    """来自同一执行者和目标者的动作请求包"""
    actor_id: str                           # 动作发送者 ID
    channel: Channel                        # 动作通道定义
    action_calls: list[dict]                # 实际要执行的动作及参数列表，来自 LLM 的 tool_calls 列表
    action_invoke_time: float               # 动作开始时刻
    action_duration: float | None = None    # 动作执行时间间隔，None 表示尚未执行完成，Actor 处于 busy 状态
    action_results: dict[str, str] = Field(default_factory=dict)  # 动作调用结果 tool_call_id -> result

    def get_action_duration(self) -> float:
        assert self.action_duration is not None, "Action duration is not set"
        return self.action_duration


class Channel(BaseModel):
    cognitive_target: str                     # 认知的目标对象别名
    target_id: str                            # 动作接收者 ID
    budget: int | None = None                 # 观察上下文预算，默认无限预算
    allowed_actions: list[str] | None = None  # 允许的动作名称列表，默认允许所有动作
    
    def get_action(self, action_name: str, world: World) -> Action:
        """获取对象的指定动作"""
        assert self.allowed_actions is None or action_name in self.allowed_actions, f"Action name {action_name} not allowed in channel {self.allowed_actions}"
        target = world.objects[self.target_id]
        action = target.actions[action_name]
        return action
    
    def has_action(self, action_name: str, world: World) -> bool:
        return action_name in self.get_allowed_actions(world)
    
    def get_allowed_actions(self, world: World) -> list[str]:
        if self.allowed_actions:
            return self.allowed_actions
        target = world.objects[self.target_id]
        return list(target.actions.keys())


class Object(Serializable):
    """对象基类"""
    
    DEFAULT_READ_SPEED: float = 10
    DEFAULT_CAPACITY: int = 1        # 默认容量为1，即只能同时处理一个动作

    object_id: str                   # 对象 ID
    actions: dict[str, Action]       # 支持的动作字典，键为动作名称，值为动作对象
    read_speed: float                # 观察对象状态的速度，单位为 tokens/second
    capacity: int                    # 客体对象最大动作并发数，0 表示无限并发

    def __init__(self, object_id: str, *,
                 actions: list[Action] | None = None,
                 read_speed: float | None = None,
                 capacity: int | None = None,
                ):
        super().__init__()
        self.object_id = object_id
        self.actions = {action.name: action for action in (actions or [])}
        self.read_speed = read_speed or self.DEFAULT_READ_SPEED
        self.capacity = capacity or self.DEFAULT_CAPACITY
    
    @property
    def objects(self) -> dict[str, Object]:
        """当前对象所持有的子对象字典，键为子对象 ID，值为子对象"""
        return cast(dict[str, Object], self._objects)
    
    def _observe_duration(self, content: str, world: World | None = None, observer_id: str | None = None) -> float:
        """计算观察对象状态的持续时间"""
        token_count = estimate_tokens(content)
        read_speed_gain: float
        if world is None or observer_id is None:
            read_speed_gain = 1.0
        else:
            observer = world.objects[observer_id]
            if isinstance(observer, Agent):
                read_speed_gain = observer.read_speed_gain
                assert read_speed_gain > 0, f"Observer {observer_id}'s read speed gain must be greater than 0"
            else:
                read_speed_gain = 1.0
        assert self.read_speed > 0, f"Object {self.object_id}'s read speed must be greater than 0"
        return token_count / (self.read_speed * read_speed_gain)
    
    def _validate_action_packages(self, packages: list[ActionExecutionPackage]):
        """验证动作请求包满足约束"""

        # 检查调用工具名称是否在通道允许列表中
        for package in packages:
            if package.channel.allowed_actions is None:
                continue
            for tool_call in package.action_calls:
                action_name = tool_call["function"]["name"]
                if action_name not in package.channel.allowed_actions:
                    raise ValueError(f"Action {action_name} not allowed in channel {package.channel.allowed_actions}")

        # 检查 target_id 与 当前对象 object_id 是否一致
        for package in packages:
            if package.channel.target_id != self.object_id:
                raise ValueError(f"Target ID {package.channel.target_id} does not match object ID {self.object_id}")

    async def _execute_action(self, tool_call: dict, actor: Agent, world: World) -> TimedStr:
        """执行工具调用"""
        action_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        action = self.actions.get(action_name)
        if action is None:
            raise ValueError(f"Action {action_name} not found in object.actions")
        params = action.GetParams(arguments)
        result = action.execute(self, params, actor, world)
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
        assert isinstance(world, World)
        return []

    async def passive(self, packages: list[ActionExecutionPackage], world: World) -> list[ActionExecutionPackage]:
        """被动阶段 - 接收外部动作影响"""
        self._validate_action_packages(packages)
        for package in packages:
            action_duration = 0.0
            for tool_call in package.action_calls:
                actor = world.objects[package.actor_id]
                assert isinstance(actor, Agent)
                action_result = await self._execute_action(tool_call, actor, world)
                action_duration += action_result.duration
                package.action_results[tool_call["id"]] = action_result.content or ""
            package.action_duration = action_duration
        return packages

    async def observe(self, *, channel: Channel | None = None, world: World | None = None, observer_id: str | None = None) -> TimedStr:
        """观察对象状态，将被嵌入到 LLM 的上下文信息中（感知与效应马尔可夫毯均可在此实现）"""
        content = json.dumps(self.state_dict(), ensure_ascii=False)
        duration = self._observe_duration(content, world, observer_id)
        return TimedStr(duration=duration, content=content)