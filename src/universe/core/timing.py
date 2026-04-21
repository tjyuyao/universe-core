# 本系统区分如下概念：墙上时间、逻辑时间、观察时间、思考时间、动作时间。
# 墙上时间：开发者所处世界的时间，格式为 ISO datetime with timezone，用于调试和日志。
# 逻辑时间：系统内部的时间，格式为实数，用于调度时表示任务所消耗的时间资源，以及仲裁处理事件的同时性判定。
# Agent 的 _react 过程被分为观察、思考、动作三个阶段，每个阶段需要消耗一定的逻辑时间，即：
#     观察时间：Agent 观察环境的时间，由 Channel 的 observe() 方法的实现间接决定。
#     思考时间：Agent 思考环境的时间，由 Agent 的 LLM token 数量间接决定。
#     动作时间：Agent 执行动作的时间，由 Action 的 execute() 方法的实现间接决定。

from pydantic import BaseModel, Field


class TimedStr(BaseModel):
    duration: float = Field(description="动作执行时间")
    content: str | None = Field(default=None, description="动作执行结果")
