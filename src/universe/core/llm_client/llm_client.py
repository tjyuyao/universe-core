"""LLM 客户端 - 封装 OpenAI API 调用

支持多 LLM 预加载和基于 llm_name 的快速切换。
支持自动文件缓存，相同请求返回相同响应。
"""

import json
from typing import Any, TypedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

from ..meta.singleton import SingletonMeta
from ..config import Config, LLMConfig
from .llm_logger import LLMLogger
from .llm_cache import LLMCache
from .validator import ToolArgumentsValidator
from .budget import estimate_tokens


class ToolCall(TypedDict):
    """工具调用请求"""
    name: str
    arguments: dict


@dataclass
class LLMResult:
    """LLM 生成的结果

    Attributes:
        thought: Agent 的想法/思考过程
        tool_calls: 工具调用列表
        log_file: 日志文件路径（可选）
        duration: 调用持续逻辑时间
    """
    thought: str
    tool_calls: list[ToolCall]
    log_file: Path | None = None
    duration: float = 0.0


class LLMClient(metaclass=SingletonMeta):

    CACHE_MAX_SIZE: int = 128
    CACHE_DIR: Path | str = "~/.universe/llm_cache"
    LOG_DIR: Path | str = "~/.universe/llm_logs"
    MAX_LOG_SESSIONS: int = 10
    BASE_THINK_SPEED: int = 15

    def __init__(self, config: Config | None = None):
        if config is None:
            config = Config()
        self._config = config
        self._clients: dict[str, AsyncOpenAI] = {}
        self._cache = LLMCache(max_size=self.CACHE_MAX_SIZE, cache_dir=Path(self.CACHE_DIR))
        self._logger = LLMLogger(Path(self.LOG_DIR), max_log_sessions=self.MAX_LOG_SESSIONS)
        self._validator = ToolArgumentsValidator()

    def _get_client(self, model: str | None = None) -> AsyncOpenAI:
        """获取或创建 AsyncOpenAI 客户端"""
        if model is None:
            model = self._config.enabled_llms[0]

        if model not in self._config.enabled_llms:
            raise ValueError(f"LLM {model} is not enabled in the config.")

        if model not in self._clients:
            llm_config = self._config.get_llm_config(model)
            self._clients[model] = AsyncOpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
            )
        return self._clients[model]

    def _build_messages(self, system_prompt: str | None = None, user_prompt: str | None = None) -> list[dict[str, str]]:
        """构建固定格式的消息列表"""

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        return messages

    async def complete(
        self,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
        think_speed_gain: float = 1.0,
    ) -> LLMResult:
        """完成一个 LLM 生成"""

        messages = self._build_messages(system_prompt, user_prompt)
        llm_config: LLMConfig = self._config.get_llm_config(model)
        model_name = llm_config.model
        timestamp = datetime.now()

        # 尝试从缓存获取
        cache_hit = False
        if self._cache is not None:
            cached = self._cache.get(
                model=model_name,
                messages=messages,
                temperature=temperature,
                tools=tools,
            )
            if cached is not None:
                cache_hit = True
                response_data = cached
            else:
                cache_hit = False

        # 缓存未命中，调用 API
        if not cache_hit:
            params: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
            }

            if tools:
                params["tools"] = tools

            result = await self._get_client(model).chat.completions.create(**params)
            response_data = result.model_dump()

            # 写入缓存
            if self._cache is not None:
                self._cache.set(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    tools=tools,
                    response=response_data,
                )

        # 生成日志文件
        log_file = self._logger.write_request_log(
            timestamp=timestamp,
            config=llm_config,
            messages=messages,
            tools=tools,
            response_data=response_data,
            cache_hit=cache_hit,
        )

        # 解析响应
        message: dict[str, Any] = response_data["choices"][0]["message"]
        tool_calls: list[ToolCall] = [tc['function'] for tc in message.get("tool_calls") or []]

        # 校验与修复工具调用参数
        if tools:
            # 验证并修复每个 tool_call 的参数
            for tool_call in tool_calls:
                arguments = tool_call["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                tool_name = tool_call["name"]
                tool_schema = self._find_tool_schema(tools, tool_name)
                if tool_schema:
                    arguments = self._validator.validate(arguments, tool_schema)

                tool_call["arguments"] = arguments

        thought = message.get("content", "")
        think_speed = self.BASE_THINK_SPEED * think_speed_gain
        duration = estimate_tokens(thought, model_name) / think_speed

        return LLMResult(
            thought=thought,
            tool_calls=tool_calls,
            log_file=log_file,
            duration=duration,
        )

    def _find_tool_schema(self, tools: list[dict[str, Any]], tool_name: str) -> dict | None:
        """从 tools 列表中找到指定 tool 的 schema

        Args:
            tools: 工具定义列表
            tool_name: 工具名称

        Returns:
            工具的 parameters schema，如果找不到返回 None
        """
        for tool in tools:
            func: dict[str, Any] = tool.get("function", {})
            if func.get("name") == tool_name:
                return func.get("parameters")
        return None
