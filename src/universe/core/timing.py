# 本系统区分如下概念：墙上时间、逻辑时间、观察时间、思考时间、动作时间。
# 墙上时间：开发者所处世界的时间，格式为 ISO datetime with timezone，用于调试和日志。
# 逻辑时间：系统内部的时间，格式为整数，用于调度时表示任务所消耗的时间资源，以及仲裁处理事件的同时性判定。
# Agent 的 _react 过程被分为观察、思考、动作三个阶段，每个阶段需要消耗一定的逻辑时间，即：
#     观察时间：Agent 观察环境的时间，由 Channel 的 observe() 方法的实现间接决定。
#     思考时间：Agent 思考环境的时间，由 Agent 的 LLM token 数量间接决定。
#     动作时间：Agent 执行动作的时间，由 Action 的 execute() 方法的实现间接决定。

from pydantic import BaseModel, Field
from datetime import datetime


class Timing(BaseModel):
    start_time: int = Field(description="过程开始时间，逻辑时间")
    duration: int = Field(default=1, description="过程持续时间，逻辑时间")
    wall_time: datetime = Field(default_factory=datetime.now, description="日志用墙上时间")

    @property
    def end_time(self) -> int:
        return self.start_time + self.duration
    
    @classmethod
    def union(cls, results: list[Timing]) -> Timing:
        """合并多个 Timing 实例，返回一个包含所有结果的 Timing 实例"""
        start_time = min(result.start_time for result in results)
        end_time = max(result.end_time for result in results)
        return cls(
            start_time=start_time,
            duration=end_time - start_time,
            wall_time=datetime.now()
        )


class TimedStr(BaseModel):
    duration: int = Field(description="动作执行时间")
    content: str | None = Field(default=None, description="动作执行结果")


class ReactTimings(BaseModel):
    observe: Timing = Field(description="观察时间")
    think: Timing = Field(description="思考时间")
    action: Timing = Field(description="动作时间")