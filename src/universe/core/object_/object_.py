import hjson  # type: ignore
from typing import TYPE_CHECKING, Type, Generic, TypeVar, cast
from pydantic import BaseModel, Field
from enum import Enum

from ..meta.generics_ import GenericsMeta
from ..timing import TimedStr
from ..llm_client import estimate_tokens, ToolCall
from .serializable import Serializable
from .state import State


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
        return cls.GetParamsType().model_validate(arguments)

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

    async def execute(self, obj: O, params: P, actor: Agent, world: World) -> TimedStatus:
        raise NotImplementedError("Subclasses of Action must implement execute()")


class ActionExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAIL = "fail"


class TimedStatus(BaseModel):
    duration: float
    status: ActionExecutionStatus


class ActionExecutionContext(BaseModel):
    """动作执行上下文"""
    tool_call: ToolCall              # LLM 工具调用请求，包含函数名、参数等
    start_time: float | None = None  # 动作开始时间，None 表示未开始
    duration: float = 0              # 动作执行时间，单位：逻辑秒
    status: ActionExecutionStatus = ActionExecutionStatus.PENDING  # 动作调用实时状态

    @property
    def end_time(self) -> float:
        assert self.start_time is not None
        return self.start_time + self.duration

    def is_finished(self) -> bool:
        return self.status in [ActionExecutionStatus.SUCCESS, ActionExecutionStatus.FAIL]

    def is_finished_at(self, time: float) -> bool:
        if self.start_time is None:
            return False
        return time - self.start_time >= self.duration

    def set_running(self) -> None:
        self.status = ActionExecutionStatus.RUNNING


class Activity(BaseModel):
    """来自同一执行者和目标者的动作请求包"""
    actor_id: str                           # 动作发送者 ID
    channel: Channel                        # 动作通道定义
    action_invoke_time: float               # 动作包发出时刻
    action_contexts: dict[str, ActionExecutionContext] = Field(default_factory=dict)  # 动作执行上下文， uuid -> context

    async def transit(self, obj: Object, world: World) -> bool:
        """执行到当前时间点

        Returns:
            done(bool): 到世界时间为止，是否足以完成所有动作
        """
        # Return Case 1: Action scheduled in the future, i.e. not even started, so not done (False).
        if world.time < self.action_invoke_time:
            return False

        actor = world.agents[self.actor_id]
        assert isinstance(actor, Agent)

        # Now execute the actions that are not finished yet until the current time.
        busy_until = self.action_invoke_time
        for context in self.action_contexts.values():
            # Skip already finished context.
            if context.is_finished():
                # Update busy_until to the previous context's end time.
                busy_until = context.end_time
                continue
            # Mark the start time of the current context. (consume busy_until)
            context.start_time = busy_until
            # Set the status to running.
            context.set_running()
            # Execute the action.
            action, params = obj.tool_call_as_action(context.tool_call)
            result = await action.execute(obj, params, actor, world)
            # Update the duration of the current context. (previous None)
            context.duration = result.duration
            # Set the status of the current context. (previous None)
            context.status = result.status
            # Return Case 2: Action not finished yet, so not done (False).
            if not context.is_finished_at(world.time):
                return False
        # Return Case 3: All actions are finished, so done (True).
        return True

    @property
    def busy_until(self) -> float:
        busy_until = self.action_invoke_time
        for context in self.action_contexts.values():
            if context.is_finished():
                busy_until = context.end_time
                continue
            break
        return busy_until


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

    object_id: State[str]              # 对象 ID
    actions: dict[str, Action]         # 支持的动作字典，键为动作名称，值为动作对象
    read_speed: State[float]           # 观察对象状态的速度，单位为 tokens/second
    activities: State[list[Activity]]  # 等待执行的动作请求包
    _busy_until: State[float]          # 最后一次执行动作的结束时间

    def __init__(self, object_id: str, *,
                 actions: list[Action] | None = None,
                 read_speed: float | None = None,
                ):
        super().__init__()
        self.object_id = object_id
        self.actions = {action.name: action for action in (actions or [])}
        self.read_speed = read_speed or self.DEFAULT_READ_SPEED
        self.activities = []
        self._busy_until = 0.0

    @property
    def objects(self) -> dict[str, Object]:
        """当前对象所持有的子对象字典，键为子对象 ID，值为子对象"""
        return cast(dict[str, Object], self._objects)

    @property
    def busy_until(self) -> float:
        return self._busy_until

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

    def _validate_action_package(self, package: Activity):
        """验证动作请求包满足约束"""

        # 检查调用工具名称是否在通道允许列表中
        if package.channel.allowed_actions is not None:
            for context in package.action_contexts.values():
                action_name = context.tool_call["name"]
                if action_name not in package.channel.allowed_actions:
                    raise ValueError(f"Action {action_name} not allowed in channel {package.channel.allowed_actions}")

        # 检查 target_id 与 当前对象 object_id 是否一致
        if package.channel.target_id != self.object_id:
            raise ValueError(f"Target ID {package.channel.target_id} does not match object ID {self.object_id}")

    def is_busy_at(self, time: float) -> bool:
        return self._busy_until > time

    def tool_call_as_action(self, tool_call: ToolCall) -> tuple[Action, Params]:
        action_name = tool_call["name"]
        arguments = tool_call["arguments"]
        action = self.actions.get(action_name)
        if action is None:
            raise ValueError(f"Action {action_name} not found in Object {self.object_id}")
        params = action.GetParams(arguments)
        return action, params

    def enqueue_activity(self, activity: Activity) -> None:
        """添加动作请求包到队列"""
        self._validate_action_package(activity)
        # Insert sorted by action_invoke_time
        for search_index, search_activity in enumerate(self.activities):
            if search_activity.action_invoke_time > activity.action_invoke_time:
                self.activities.insert(search_index, activity)
                break
        else:
            self.activities.append(activity)

    async def transit(self, world: World) -> None:
        """对象状态转移"""
        if self.activities:
            busy_until = self.activities[0].busy_until
            while self.activities:
                self.activities[0].action_invoke_time = busy_until
                done = await self.activities[0].transit(self, world)
                busy_until = self.activities[0].busy_until
                if done:
                    # Action finished, so pop it from the list. (drop it)
                    self.activities.pop(0)
                else:
                    # Action timeout, leave for next time.
                    break
            self._busy_until = busy_until

    async def observe(self, *, channel: Channel | None = None, world: World, observer_id: str | None = None) -> TimedStr:
        """观察对象状态，将被嵌入到 LLM 的上下文信息中（感知与效应马尔可夫毯均可在此实现）"""
        content = hjson.dumps(self.state_dict(), ensure_ascii=False)
        duration = self._observe_duration(content, world, observer_id)
        return TimedStr(duration=duration, content=content)

    async def act(self, activity: Activity) -> None:
        """执行动作请求包"""
        self.enqueue_activity(activity)
