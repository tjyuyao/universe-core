"""配置管理 - 加载和解析 config.yml"""

from dataclasses import dataclass
from typing import Any
from .singleton import SingletonMeta

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
class Config(metaclass=SingletonMeta):
    """应用配置"""
    
    DEFAULT_PATH = "config.yml"
    
    api_pool: list[LLMConfig]
    enabled_llms: list[str]

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        """从 YAML 文件加载配置

        Args:
            path: 配置文件路径，默认为项目根目录的 config.yml

        Returns:
            Config 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.api_pool = [LLMConfig.from_dict(item) for item in data.get("api_pool", [])]
        self.enabled_llms = data.get("enabled_llms", [])

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
