import time
import random

from typing import cast
from ..timing import TEPS
from ..object_ import Serializable, Object, Activity
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
        """世界步进（自适应步长）

        每步随机化 Agent 执行顺序，确保资源竞争时的公平性。
        """

        self._time += TEPS

        # 随机化 Agent 执行顺序，避免固定顺序导致的抢占式不公平
        agents = list(self.agents.values())
        random.shuffle(agents)

        for agent in agents:
            await agent.react(self)

        # 更新时间
        busy_times = []
        for obj in self.objects.values():
            if isinstance(obj, Agent):
                busy_times.append(obj.busy_until)
        if busy_times:
            self._time = min(busy_times)
