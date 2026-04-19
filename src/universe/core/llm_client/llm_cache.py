"""LLM 缓存模块 - 文件持久化的 LRU 缓存

提供基于文件系统的 LLM 响应缓存，支持：
- SHA256 哈希键生成
- LRU 淘汰策略（默认 128 条）
- 用户家目录存储（~/.ganttworld/llm_cache/）
- 异步安全的读写操作
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    """缓存条目

    Attributes:
        key_hash: 缓存键的 SHA256 哈希
        request: 原始请求数据
        response: LLM 响应数据
        created_at: 创建时间戳（ISO 格式）
        accessed_at: 最后访问时间戳（ISO 格式）
        access_count: 访问次数
    """
    key_hash: str
    request: dict[str, Any]
    response: dict[str, Any]
    created_at: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())
    accessed_at: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())
    access_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "key_hash": self.key_hash,
            "request": self.request,
            "response": self.response,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        """从字典创建"""
        return cls(
            key_hash=data["key_hash"],
            request=data["request"],
            response=data["response"],
            created_at=data.get("created_at", ""),
            accessed_at=data.get("accessed_at", ""),
            access_count=data.get("access_count", 0),
        )

    def touch(self) -> None:
        """更新访问时间和计数"""
        from datetime import datetime
        self.accessed_at = datetime.now().isoformat()
        self.access_count += 1


class LLMCache:
    """LLM 文件缓存

    使用 LRU 策略管理缓存条目，持久化存储在用户家目录。

    Example:
        >>> cache = LLMCache(max_size=128)
        >>> cache.get(model, messages, temperature)  # 获取缓存
        >>> cache.set(model, messages, temperature, response)  # 设置缓存
    """

    DEFAULT_MAX_SIZE: int = 128
    CACHE_DIR_NAME: str = ".universe/llm_cache"
    INDEX_FILE: str = "index.json"

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE, cache_dir: Path | None = None):
        """初始化缓存

        Args:
            max_size: 最大缓存条目数，默认 128
            cache_dir: 缓存目录，默认 ~/.universe/llm_cache/
        """
        self._max_size = max_size
        self._cache_dir = cache_dir or self._get_default_cache_dir()
        self._index: OrderedDict[str, str] = OrderedDict()  # hash -> filepath

        self._ensure_cache_dir()
        self._load_index()

    @classmethod
    def _get_default_cache_dir(cls) -> Path:
        """获取默认缓存目录"""
        home = Path.home()
        return home / cls.CACHE_DIR_NAME

    def _ensure_cache_dir(self) -> None:
        """确保缓存目录存在"""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_index_path(self) -> Path:
        """获取索引文件路径"""
        return self._cache_dir / self.INDEX_FILE

    def _get_cache_file_path(self, key_hash: str) -> Path:
        """获取缓存文件路径

        使用前缀分目录避免单目录文件过多：
        - hash: abcdef123456... -> cache_dir/ab/abcdef123456....json
        """
        prefix = key_hash[:2]
        subdir = self._cache_dir / prefix
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key_hash}.json"

    def _load_index(self) -> None:
        """从磁盘加载索引"""
        index_path = self._get_index_path()
        if not index_path.exists():
            return

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key_hash in data.get("order", []):
                filepath = self._get_cache_file_path(key_hash)
                if filepath.exists():
                    self._index[key_hash] = str(filepath)

        except (json.JSONDecodeError, IOError):
            pass

    def _save_index(self) -> None:
        """保存索引到磁盘"""
        index_path = self._get_index_path()
        data = {"order": list(self._index.keys())}

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _generate_key(
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        tools: list[dict[str, Any]] | None,
    ) -> str:
        """生成缓存键

        使用 SHA256 哈希确保相同请求产生相同键。

        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 采样温度
            max_tokens: 最大 token 数
            tools: 工具定义

        Returns:
            SHA256 哈希字符串
        """
        key_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
        }
        key_json = json.dumps(key_data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(key_json.encode("utf-8")).hexdigest()

    def _evict_if_needed(self) -> None:
        """如果需要，执行 LRU 淘汰"""
        while len(self._index) >= self._max_size:
            oldest_hash, oldest_path = self._index.popitem(last=False)
            try:
                Path(oldest_path).unlink(missing_ok=True)
            except OSError:
                pass

    def get(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """获取缓存

        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 采样温度
            max_tokens: 最大 token 数
            tools: 工具定义

        Returns:
            缓存的响应数据，未命中返回 None
        """
        key_hash = self._generate_key(model, messages, temperature, tools)

        if key_hash not in self._index:
            return None

        filepath = Path(self._index[key_hash])

        if not filepath.exists():
            del self._index[key_hash]
            self._save_index()
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            entry = CacheEntry.from_dict(data)
            entry.touch()

            # 更新访问时间到文件
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)

            # 移动到末尾（最近使用）
            self._index.move_to_end(key_hash)
            self._save_index()

            return entry.response

        except (json.JSONDecodeError, IOError, KeyError):
            return None

    def set(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        tools: list[dict[str, Any]] | None,
        response: dict[str, Any],
    ) -> None:
        """设置缓存

        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 采样温度
            tools: 工具定义
            response: LLM 响应数据
        """
        key_hash = self._generate_key(model, messages, temperature, tools)

        # 如果已存在，先删除旧文件
        if key_hash in self._index:
            old_path = Path(self._index[key_hash])
            old_path.unlink(missing_ok=True)
            del self._index[key_hash]

        # 执行淘汰
        self._evict_if_needed()

        # 创建新条目
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
        }
        entry = CacheEntry(
            key_hash=key_hash,
            request=request_data,
            response=response,
        )

        filepath = self._get_cache_file_path(key_hash)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)

        self._index[key_hash] = str(filepath)
        self._save_index()

    def clear(self) -> int:
        """清空所有缓存

        Returns:
            删除的条目数
        """
        count = 0
        for filepath in self._index.values():
            try:
                Path(filepath).unlink(missing_ok=True)
                count += 1
            except OSError:
                pass

        self._index.clear()
        self._save_index()
        return count

    def stats(self) -> dict[str, Any]:
        """获取缓存统计信息

        Returns:
            统计信息字典
        """
        total_size = 0
        for filepath in self._index.values():
            try:
                total_size += Path(filepath).stat().st_size
            except OSError:
                pass

        return {
            "entry_count": len(self._index),
            "max_size": self._max_size,
            "cache_dir": str(self._cache_dir),
            "total_size_bytes": total_size,
        }
