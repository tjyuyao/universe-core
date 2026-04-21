from typing import Any


class SingletonMeta(type):
    """单例元类"""
    
    _instance: Any
    
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance
