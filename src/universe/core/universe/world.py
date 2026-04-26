import time
import random

from typing import cast
from ..timing import TEPS, TINF
from ..object_ import Serializable, Object, PrivateState
from ..agent import Agent



class World(Serializable):
    """世界基类"""

    name: str
    description: str
    _time: PrivateState[float]  # 当前时间 (private, 会序列化但不参与观察)
    _init_time: float             # 世界开始时间 (const)
    _random: random.Random        # 随机数生成器（普通属性，不参与序列化）

    def __init__(self, name: str, description: str, init_time: float | None = None, random_seed: int = 42) -> None:
        super().__init__()
        self.name = name
        self.description = description
        self._time = 0
        self._init_time = init_time or time.time()
        self._random = random.Random(random_seed)

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
        self._random.shuffle(agents)

        for agent in agents:
            await agent.react(self)

        # 更新时间
        busy_times = [obj.busy_until for obj in self.agents.values()]
        if busy_times:
            self._time = min(busy_times)
                
    async def finalize(self, time: float = TINF) -> None:
        """完成世界运行，确保所有 Agent 都完成其任务。"""
        
        # 随机化 Agent 执行顺序，避免固定顺序导致的抢占式不公平
        agents = list(self.agents.values())
        self._random.shuffle(agents)
        
        # 更新时间
        self._time = self._time + time
        
        for agent in agents:
            await agent.react(self, final=True)

    def state_dict(self) -> dict:
        """返回包含随机状态的世界状态字典"""
        state = super().state_dict()
        state["_random_state"] = self._random.getstate()
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """从状态字典加载世界状态（包括随机状态）"""
        # 提取并移除随机状态，避免父类处理未知键
        random_state = state_dict.pop("_random_state", None)
        super().load_state_dict(state_dict)
        if random_state is not None:
            self._random.setstate(random_state)
