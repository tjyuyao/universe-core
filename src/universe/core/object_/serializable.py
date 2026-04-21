import hjson as json  # type: ignore
import warnings
from pydantic_core import PydanticSerializationError
from pydantic import BaseModel
from typing import Any, ClassVar, get_type_hints
from ..translate import translate
from .state import BaseState, is_state_annotation


# Type alias for valid state value types
StateValue = int | float | str | bool | None | list | tuple | dict | BaseModel | BaseState


class Serializable:
    """对象基类"""

    _objects: dict[str, Serializable]
    """持有的对象字典，键为属性名，值为对象实例"""

    _states: dict[str, StateValue]
    """对象的状态字典，键为属性名，值为属性值"""

    _state_fields_: ClassVar[set[str]] = set()
    """类级别的状态字段集合，通过 State[T] 注解自动收集"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Collect field names with State[T] annotations
        try:
            hints = get_type_hints(cls, include_extras=True)
            cls._state_fields_ = {
                name for name, hint in hints.items() if is_state_annotation(hint)
            }
        except NameError:
            warnings.warn(
                f"Failed to resolve type hints for {cls.__name__}, _state_fields_ will be empty",
                UserWarning,
            )
            cls._state_fields_ = set()

    def __init__(self):
        self._objects = {}
        self._states = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(states={list(self._states.keys())}, objects={list(self._objects.keys())})"

    def __setattr__(self, name: str, value: Any):
        if name in ("_objects", "_states"):
            super().__setattr__(name, value)
            return

        assert hasattr(self, "_objects")
        assert hasattr(self, "_states")

        if isinstance(value, Serializable):
            self.register_object(name, value)
        elif name in type(self)._state_fields_:
            self.register_state(name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if name in self._states:
            return self._states[name]
        if name in self._objects:
            return self._objects[name]
        raise AttributeError(translate(f"未定义属性 '{name}'"))

    def _backref_detect(self, obj: Serializable, seen: set | None = None) -> bool:
        if self is obj:
            return True
        if seen is None:
            seen = set()
        seen.add(id(self))
        if id(obj) in seen:
            return True
        for child in self._objects.values():
            if child._backref_detect(obj, seen):
                return True
        return False

    @property
    def states(self) -> dict[str, StateValue]:
        return self._states

    @staticmethod
    def serialize_state(state: StateValue, name: str) -> Any:
        """序列化状态变量到字典"""
        if isinstance(state, (int, float, str, bool, type(None))):
            return state
        elif isinstance(state, (list, tuple, dict)):
            try:
                json.dumps(state, ensure_ascii=False)
            except TypeError:
                raise TypeError(translate(f"状态变量 '{name}' 不可序列化"))
            return state
        elif isinstance(state, BaseModel):
            try:
                data = state.model_dump()
                json.dumps(data, ensure_ascii=False)
            except (TypeError, PydanticSerializationError):
                raise TypeError(translate(f"状态变量 '{name}' 不可序列化"))
            return data
        elif isinstance(state, BaseState):
            try:
                data = state.model_dump()
                json.dumps(data, ensure_ascii=False)
            except (TypeError, PydanticSerializationError):
                raise TypeError(translate(f"状态变量 '{name}' 不可序列化"))
            return data
        else:
            raise TypeError(translate(f"未支持的状态变量类型 '{type(state)}'"))

    def state_dict(self) -> dict[str, Any]:
        """返回对象的状态字典"""
        destination: dict[str, Any] = {}

        for name, state in self._states.items():
            destination[name] = Serializable.serialize_state(state, name)

        for name, value in self._objects.items():
            destination[name] = value.state_dict()
        return destination

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """从状态字典加载对象的状态"""
        used_keys: set[str] = set()

        for name, state in self._states.items():
            if name not in state_dict:
                continue

            if isinstance(state, (int, float, str, bool, type(None))):
                self._states[name] = state_dict[name]
            elif isinstance(state, (list, tuple, dict)):
                self._states[name] = state_dict[name]
            elif isinstance(state, BaseModel):
                self._states[name] = state.model_validate(state_dict[name])
            elif isinstance(state, BaseState):
                self._states[name] = state.model_validate(state_dict[name])
            else:
                raise TypeError(translate(f"未支持的状态变量类型 '{type(state)}'"))
            used_keys.add(name)

        for name in self._objects.keys():
            if name in state_dict:
                self._objects[name].load_state_dict(state_dict[name])
                used_keys.add(name)

        for name in state_dict.keys():
            if name not in used_keys:
                warnings.warn(translate(f"状态字典中未使用键 '{name}'"), UserWarning)

    def register_object(self, name: str, obj: Serializable) -> None:
        """注册一个对象"""
        if name in self._objects:
            warnings.warn(translate(f"对象 '{name}' 已存在，将被覆盖"), UserWarning)
        if name in self._states:
            raise ValueError(translate(f"不允许同名的对象与状态变量 '{name}'"))
        if self._backref_detect(obj):
            raise ValueError(translate(f"对象 '{name}' 存在循环引用"))
        self._objects[name] = obj

    def register_state(self, name: str, state: StateValue) -> None:
        """注册一个状态变量"""
        if name in self._objects:
            raise ValueError(translate(f"不允许同名的状态变量与对象 '{name}'"))

        # 检查状态变量是否可序列化
        if isinstance(state, (int, float, str, bool, type(None))):
            pass
        elif isinstance(state, (list, tuple, dict)):
            try:
                json.dumps(state, ensure_ascii=False)
            except TypeError:
                raise TypeError(translate(f"状态变量 '{name}' 不可序列化"))
        elif isinstance(state, BaseModel):
            try:
                data = state.model_dump()
                json.dumps(data, ensure_ascii=False)
            except (TypeError, PydanticSerializationError):
                raise TypeError(translate(f"状态变量 '{name}' 不可序列化"))
        elif isinstance(state, BaseState):
            try:
                data = state.model_dump()
                json.dumps(data, ensure_ascii=False)
            except (TypeError, PydanticSerializationError):
                raise TypeError(translate(f"状态变量 '{name}' 不可序列化"))
        else:
            raise TypeError(translate(f"未支持的状态变量类型 '{type(state)}'"))

        self._states[name] = state
