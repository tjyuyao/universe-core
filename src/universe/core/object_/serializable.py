import json
from typing import Any
from ..translate import translate
from .state import State
import warnings


class Serializable:
    """对象基类"""
    
    _objects: dict[str, Serializable]
    """持有的对象字典，键为属性名，值为对象实例"""
    
    _states: dict[str, State]
    """对象的状态字典，键为属性名，值为属性值"""
       
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
            if name in self._objects:
                warnings.warn(translate(f"对象 '{name}' 已存在，将被覆盖"), UserWarning)
            if name in self._states:
                raise ValueError(translate(f"不允许同名的对象与状态变量 '{name}'"))
            if self._backref_detect(value):
                raise ValueError(translate(f"对象 '{name}' 存在循环引用"))
            self._objects[name] = value
        elif isinstance(value, State):
            if name in self._objects:
                raise ValueError(translate(f"不允许同名的状态变量与对象 '{name}'"))
            try:
                json.dumps(value, ensure_ascii=False)
            except TypeError:
                raise ValueError(translate(f"状态变量 '{name}' 不可序列化"))
            self._states[name] = value
        
        super().__setattr__(name, value)

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

    def state_dict(self) -> dict[str, Any]:
        """返回对象的状态字典"""
        destination: dict[str, Any] = {}
        destination.update(self._states)
        for name, value in self._objects.items():
            destination[name] = value.state_dict()
        return destination
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """从状态字典加载对象的状态"""
        used_keys: set[str] = set()
        
        for name in self._states.keys():
            if name in state_dict:
                self._states[name] = state_dict[name]
                used_keys.add(name)
        
        for name in self._objects.keys():
            if name in state_dict:
                self._objects[name].load_state_dict(state_dict[name])
                used_keys.add(name)
                
        for name in state_dict.keys():
            if name not in used_keys:
                warnings.warn(translate(f"状态字典中未使用键 '{name}'"), UserWarning)