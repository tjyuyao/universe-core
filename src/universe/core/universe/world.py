from ..object_ import Object


class World(Object):
    """世界基类"""
    
    name: str
    description: str
    _time: int  # 当前时间
    
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._time = 0

    @property
    def time(self) -> int:
        return self._time    