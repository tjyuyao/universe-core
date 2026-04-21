import time

from typing import cast
from ..object_ import Serializable, Object, ActionExecutionPackage
from ..agent import Agent



class World(Serializable):
    """世界基类"""

    name: str
    description: str
    _time: float             # 当前时间 (private)
    _init_time: float        # 世界开始时间 (const)

    def __init__(self, name: str, description: str, init_time: float | None = None) -> None:
        super().__init__()
        self.name = name
        self.description = description
        self._time = 0
        self._init_time = init_time or time.time()

    @property
    def time(self) -> float:
        return self._time
    
    @property
    def epoch_time(self) -> float:
        return self._init_time
    
    @property
    def objects(self) -> dict[str, Object]:
        return cast(dict[str, Object], self._objects)
    
    @property
    def agents(self) -> dict[str, Agent]:
        return {obj.object_id: obj for obj in self.objects.values() if isinstance(obj, Agent)}
        
    async def step(self) -> None:
        """世界步进（自适应步长）"""
        
        self._time += 1e-3
        
        # 主动阶段
        router: dict[str, list[ActionExecutionPackage]] = {}
        """target_id -> packages"""
        for obj in self.objects.values():
            assert isinstance(obj, Object)
            active_packages = await obj.active(self)
            for pkg in active_packages:
                router.setdefault(pkg.channel.target_id, []).append(pkg)
    
        # 被动阶段
        for obj in self.objects.values():
            object_packages = await obj.passive(router[obj.object_id], self)
            for pkg in object_packages:
                actor = self.objects[pkg.actor_id]
                assert isinstance(actor, Agent)
                actor.append_busy_time(pkg.get_action_duration())
        
        # 更新时间
        busy_times = []
        for obj in self.objects.values():
            if isinstance(obj, Agent):
                busy_times.append(obj.busy_until)
        self._time = min(busy_times)
