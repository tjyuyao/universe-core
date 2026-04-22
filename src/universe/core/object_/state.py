from typing import Annotated, Any, Self, get_origin, get_args


class _StateMarker:
    pass


_STATE_MARKER = _StateMarker()


class _PrivateStateMarker:
    pass


_PRIVATE_STATE_MARKER = _PrivateStateMarker()


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


type PrivateState[T] = Annotated[T, _PRIVATE_STATE_MARKER]
"""
Use PrivateState[T] in annotations to mark fields as serialized-only state.
These fields are included in state_dict() for save/load but excluded from observation.

Example:
    class MyClass(Serializable):
        name: State[str]           # observable by others
        secret: PrivateState[str]  # hidden from observation
"""


def is_state_annotation(hint: Any) -> bool:
    """Check if a type hint is a State[T] annotation."""
    # Python 3.12+ type alias: State[T] = Annotated[T, _STATE_MARKER]
    # get_type_hints() returns State[str] (GenericAlias with origin State),
    # NOT Annotated[str, _STATE_MARKER]. The _STATE_MARKER is lost in resolution.
    # This is the primary path for Python 3.12+.
    if get_origin(hint) is State:
        return True
    # Fallback: direct Annotated usage (if someone writes Annotated[T, _STATE_MARKER], which is unlikely though)
    if get_origin(hint) is Annotated:
        args = get_args(hint)
        return any(isinstance(a, _StateMarker) for a in args[1:])
    return False


def is_private_state_annotation(hint: Any) -> bool:
    """Check if a type hint is a PrivateState[T] annotation."""
    # Python 3.12+ type alias: PrivateState[T] = Annotated[T, _PRIVATE_STATE_MARKER]
    # Same behavior as State — get_type_hints() returns PrivateState[str] (origin PrivateState).
    # This is the primary path for Python 3.12+.
    if get_origin(hint) is PrivateState:
        return True
    # Fallback: direct Annotated usage
    if get_origin(hint) is Annotated:
        args = get_args(hint)
        return any(isinstance(a, _PrivateStateMarker) for a in args[1:])
    return False


__all__ = ["State", "PrivateState", "BaseState", "is_state_annotation", "is_private_state_annotation"]
