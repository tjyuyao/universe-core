from dataclasses import dataclass
from typing import Type, Generic, TypeVar
from pydantic import BaseModel

from ..meta.generics_ import GenericsMeta
from .serializable import Serializable


O = TypeVar("O", bound="Object")
P = TypeVar("P", bound="Params")
B = TypeVar("B")  # Global Broadcast


class Params(BaseModel):
    pass


class Action(Generic[O, P, B], metaclass=GenericsMeta):

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

    def execute(self, obj: O, params: P, globals_: B) -> str:
        raise NotImplementedError("Subclasses of Action must implement execute()")


class ActionExecutionPackage(BaseModel):
    """来自同一执行者和目标者的动作请求包"""
    channel: Channel           # 动作通道定义
    actor_id: str              # 动作发送者 ID
    target_id: str             # 动作接收者 ID
    tool_calls: list[dict]     # 实际要执行的 tool_call 列表
    tool_call_results: dict[str, str]  # 工具调用结果列表 tool_call_id -> result
    # TODO: add time start and duration


class Channel(BaseModel):
    allowed_actions: list[str] | None = None  # 允许的工具调用名称列表，默认允许所有工具


class Object(Generic[B], Serializable):
    """对象基类"""

    object_id: str
    actions: dict[str, Action]

    def __init__(self, object_id: str, actions: list[Action] | None = None):
        super().__init__()
        self.object_id = object_id
        self.actions = {action.name: action for action in (actions or [])}

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
            if package.target_id != self.object_id:
                raise ValueError(f"Target ID {package.target_id} does not match object ID {self.object_id}")

    async def _execute_action(self, tool_call: dict, globals_: B) -> str:
        """执行工具调用"""
        action_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        action = self.actions.get(action_name)
        if action is None:
            raise ValueError(f"Action {action_name} not found in object.actions")
        params = action.GetParams(arguments)
        result = action.execute(self, params, globals_)
        return result

    async def observe(self, *, budget: int | None = None, channel: Channel | None = None) -> dict:
        """观察对象状态"""
        return self.state_dict()

    async def active(self, globals_: B) -> None:
        """主动阶段
         
        职责一：模拟状态随时间的自然演化
            例如：
            - 无控制状态转移
            - 惯性运动更新位置
            - 随着年龄衰老
            - 资源自然消耗
            - 过期数据清理
            
        职责二：向其他对象发送动作请求
        """
        pass

    async def passive(self, packages: list[ActionExecutionPackage], globals_: B) -> list[ActionExecutionPackage]:
        """被动阶段 - 接收外部动作影响
        """
        packages = await self.arbitrate(packages)

        self._validate_action_packages(packages)

        for package in packages:
            for tool_call in package.tool_calls:
                action_result = await self._execute_action(tool_call, globals_)
                package.tool_call_results[tool_call["id"]] = action_result

        return packages

    async def arbitrate(self, packages: list[ActionExecutionPackage]) -> list[ActionExecutionPackage]:
        """仲裁阶段 - 处理同时发起的多个动作请求（默认透传，子类可重写）
        """
        return packages