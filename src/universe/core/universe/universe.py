from ..object_ import Serializable
from .world import World


class Universe(Serializable):
    """宇宙基类"""
    
    def __init__(self, name: str, worlds: dict[str, World] = {}):
        super().__init__()
        self.name = name
        for world_name, world in worlds.items():
            self.add_world(world_name, world)
    
    @property
    def worlds(self) -> dict[str, World]:
        return {k: v for k, v in self._objects.items() if isinstance(v, World)}
    
    def add_world(self, world_name: str, world: World):
        self.register_object(world_name, world)