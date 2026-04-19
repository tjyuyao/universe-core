from typing import Any


class SingletonMeta(type):
    """单例元类"""
    
    _instances: dict[type, Any] = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = type.__call__(cls, *args, **kwargs)
        return cls._instances[cls]
