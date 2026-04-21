import hjson as json  # type: ignore
import tiktoken  # type: ignore
from typing import Any
from pydantic import BaseModel
from ..config import Config


_config = Config()


def estimate_tokens(content: Any, model: str | None = None) -> int:
    """使用 tiktoken 估算内容的 token 数

    Args:
        content: 要估算的内容（支持 str、dict、list、BaseModel）
        model: 模型名称，用于选择对应的编码器

    Returns:
        token 数量
    """
    # 统一转换为字符串
    if isinstance(content, str):
        text = content
    elif isinstance(content, BaseModel):
        text = json.dumps(content.model_dump(), ensure_ascii=False)
    else:
        text = json.dumps(content, ensure_ascii=False)

    try:
        encoding = tiktoken.encoding_for_model(model or _config.get_llm_config().model)
    except KeyError:
        # 未知模型使用 cl100k_base（gpt-4 的编码）
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


class BudgetWarning(UserWarning):
    """当压缩后的内容超过预算时发出的警告

    Attributes:
        actual: 实际的 token 数量
        budget: 预算限制
        context_type: 上下文类型名称
    """

    def __init__(self, actual: int, budget: int, context_type: str = ""):
        self.actual = actual
        self.budget = budget
        self.context_type = context_type
        message = (
            f"Token budget exceeded for {context_type}: "
            f"{actual} > {budget} (over by {actual - budget})"
        )
        super().__init__(message)
