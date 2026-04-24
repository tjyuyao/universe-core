"""配置管理 - 加载和解析 config.yml"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from .meta.singleton import SingletonMeta

import yaml


@dataclass
class LLMConfig:
    """单个 LLM 配置"""

    name: str
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMConfig":
        """从字典创建配置"""
        return cls(
            name=data["name"],
            base_url=data["base_url"],
            api_key=data["api_key"],
            model=data["model"],
        )


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""

    name: str
    url: str
    api_key: str
    model: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingConfig":
        """从字典创建配置"""
        return cls(
            name=data["name"],
            url=data["url"],
            api_key=data["api_key"],
            model=data["model"],
        )


@dataclass
class Config(metaclass=SingletonMeta):
    """应用配置"""

    ENV_VAR: ClassVar[str] = "UNIVERSE_CONFIG_PATH"
    DEFAULT_STORAGE: ClassVar[str] = ".storage"

    api_pool: list[LLMConfig]
    enabled_llms: list[str]
    embedding_configs: list[EmbeddingConfig]
    storage: str

    def __init__(self, path: str | None = None) -> None:
        """从 YAML 文件加载配置

        配置查找顺序（优先级从高到低）：
        1. 传入的 path 参数
        2. UNIVERSE_CONFIG_PATH 环境变量
        3. ~/.universe/config.yml
        4. ./config.yml

        Args:
            path: 配置文件路径，优先使用

        Returns:
            Config 实例
        """
        config_path = self._resolve_config_path(path)

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.api_pool = [LLMConfig.from_dict(item) for item in data.get("api_pool", [])]
        self.enabled_llms = data.get("enabled_llms", [])
        self.embedding_configs = [EmbeddingConfig.from_dict(item) for item in data.get("embeddings", [])]
        self.storage = data.get("storage", self.DEFAULT_STORAGE)

    @classmethod
    def _resolve_config_path(cls, override_path: str | None = None) -> Path:
        """解析配置文件路径

        优先级：
        1. override_path 参数
        2. UNIVERSE_CONFIG_PATH 环境变量
        3. 默认路径列表中的路径

        Args:
            override_path: 手动指定的路径，最高优先级

        Returns:
            存在的配置文件路径

        Raises:
            FileNotFoundError: 找不到任何配置文件
        """
        # 默认路径列表（按优先级排序）
        default_paths = [
            "config.yml",  # 当前目录
            "${HOME}/.universe/config.yml",  # 用户主目录
        ]

        # 1. 检查传入的参数
        if override_path:
            path = Path(override_path).expanduser()
            if path.exists():
                return path
            raise FileNotFoundError(f"指定的配置文件不存在: {override_path}")

        # 2. 检查环境变量
        env_path = os.environ.get(cls.ENV_VAR)
        if env_path:
            path = Path(env_path).expanduser()
            if path.exists():
                return path
            raise FileNotFoundError(f"环境变量 {cls.ENV_VAR} 指向的配置文件不存在: {env_path}")

        # 3. 检查默认路径列表
        for path_template in default_paths:
            # 替换 ${HOME} 为实际 home 目录
            expanded_path = path_template.replace("${HOME}", str(Path.home()))
            path = Path(expanded_path).expanduser()
            if path.exists():
                return path

        # 4. 都找不到，抛出错误
        searched = [override_path, env_path] + [p.replace("${HOME}", str(Path.home())) for p in default_paths]
        raise FileNotFoundError(
            f"找不到配置文件。搜索路径:\n" + "\n".join(f"  - {p}" for p in searched if p)
        )

    def get_llm_config(self, name: str | None = None) -> LLMConfig:
        """获取 LLM 配置

        Args:
            name: LLM 名称，默认使用第一个 enabled_llm

        Returns:
            LLMConfig 实例

        Raises:
            ValueError: 找不到对应的配置
        """
        if name is None:
            if not self.enabled_llms:
                raise ValueError("No enabled LLMs found")
            name = self.enabled_llms[0]

        for config in self.api_pool:
            if config.name == name:
                return config

        raise ValueError(f"LLM config not found: {name}")

    def get_enabled_configs(self) -> list[LLMConfig]:
        """获取所有启用的 LLM 配置"""
        return [self.get_llm_config(name) for name in self.enabled_llms]

    def get_embedding_config(self, name: str) -> EmbeddingConfig:
        """获取嵌入模型配置

        Args:
            name: 嵌入配置名称

        Returns:
            EmbeddingConfig 实例

        Raises:
            ValueError: 找不到对应的配置
        """
        for config in self.embedding_configs:
            if config.name == name:
                return config

        raise ValueError(f"Embedding config not found: {name}")
