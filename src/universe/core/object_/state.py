from typing import Annotated, Any, Self, get_origin, get_args


class _StateMarker:
    pass


_STATE_MARKER = _StateMarker()


class BaseState:
    """Base class for custom serializable state types.

    Subclass this to create state types with custom serialization.
    """

    def __class_getitem__(cls, item):
        return Annotated[item, _STATE_MARKER]

    def model_dump(self) -> dict:
        """将状态转换为字典"""
        raise NotImplementedError("子类必须实现 model_dump 方法")

    @classmethod
    def model_validate(cls, state_dict: dict) -> Self:
        """从字典验证并构建状态"""
        raise NotImplementedError("子类必须实现 model_validate 方法")


type State[T] = Annotated[T, _STATE_MARKER]
"""
Use State[T] in annotations to mark fields as serializable state.

Example:
    class MyClass(Serializable):
        name: State[str]
        count: State[int]
"""


def is_state_annotation(hint: Any) -> bool:
    """Check if a type hint is a State[T] annotation."""
    if get_origin(hint) is Annotated:
        args = get_args(hint)
        return any(isinstance(a, _StateMarker) for a in args[1:])
    return False


__all__ = ["State", "BaseState", "is_state_annotation"]
