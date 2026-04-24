"""
Embedding Models - 语义嵌入模块

提供文本和图像的嵌入向量生成功能。
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Literal, Any
import httpx
import pydantic


def get_embedding_config(name: str = "doubao-vision") -> "EmbeddingConfig":
    """从全局 Config 获取嵌入模型配置

    如果 Config 中找不到配置，则返回默认的硬编码配置。

    Args:
        name: 嵌入配置名称，默认为 "doubao-vision"

    Returns:
        EmbeddingConfig 实例
    """
    try:
        from ..config import Config
        config = Config()
        embedding_config = config.get_embedding_config(name)
        return EmbeddingConfig(
            url=embedding_config.url,
            key=embedding_config.api_key,
            model=embedding_config.model,
        )
    except (ImportError, FileNotFoundError, ValueError):
        # Config 未初始化、找不到配置文件或配置不存在时，使用默认配置
        if name == "doubao-vision":
            return _DEFAULT_EMBEDDING
        raise ValueError(f"Embedding config not found: {name}")


class EmbeddingCache:
    """嵌入请求缓存管理器"""

    DEFAULT_CACHE_DIR: str = ".storage/embed_cache"

    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            # 尝试从 Config 获取 storage 路径
            try:
                from ..config import Config
                config = Config()
                cache_dir = str(Path(config.storage) / "embed_cache")
            except (ImportError, FileNotFoundError):
                cache_dir = self.DEFAULT_CACHE_DIR
        self.cache_dir = Path(cache_dir)
        self._dir_created = False

    @property
    def enabled(self) -> bool:
        """动态检查缓存是否启用（默认启用，通过环境变量关闭）"""
        return os.environ.get("EMBEDDING_CACHE", "1") != "0"

    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        if not self._dir_created and self.enabled:
            self.cache_dir.mkdir(exist_ok=True)
            self._dir_created = True

    def _compute_cache_key(self, model: str, inputs: list) -> str:
        """计算缓存键"""
        cache_data = {
            "model": model,
            "inputs": inputs,
        }
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(cache_str.encode("utf-8")).hexdigest()

    def _get_cache_file(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, model: str, inputs: list) -> dict | None:
        """获取缓存结果"""
        if not self.enabled:
            return None

        self._ensure_cache_dir()
        cache_key = self._compute_cache_key(model, inputs)
        cache_file = self._get_cache_file(cache_key)

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def set(self, model: str, inputs: list, result: dict) -> None:
        """设置缓存结果"""
        if not self.enabled:
            return

        self._ensure_cache_dir()
        cache_key = self._compute_cache_key(model, inputs)
        cache_file = self._get_cache_file(cache_key)

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except IOError:
            pass


_embedding_cache = EmbeddingCache()


class EmbeddingInput(pydantic.BaseModel):
    """嵌入输入项"""
    type: Literal["text", "image_url"] = "text"
    text: str | None = None
    image_url: dict | None = None

    def to_api_format(self) -> dict:
        """转换为 API 格式"""
        if self.type == "text":
            return {"type": "text", "text": self.text}
        else:
            return {"type": "image_url", "image_url": self.image_url}


class EmbeddingResult(pydantic.BaseModel):
    """嵌入结果"""
    index: int
    embedding: list[float]
    object: str = "embedding"


class EmbeddingResponse(pydantic.BaseModel):
    """嵌入响应"""
    object: str = "list"
    data: list[EmbeddingResult]
    model: str
    usage: dict[str, int | dict] = {}


class EmbeddingConfig(pydantic.BaseModel):
    """嵌入模型配置"""
    url: str
    key: str
    model: str

    def embed(
        self,
        inputs: list[EmbeddingInput] | list[str] | list[dict],
        use_cache: bool = True,
    ) -> EmbeddingResponse:
        """生成嵌入向量

        Args:
            inputs: 输入列表，可以是 EmbeddingInput 对象、字符串或字典
            use_cache: 是否使用缓存

        Returns:
            EmbeddingResponse 包含嵌入向量结果
        """
        normalized_inputs = []
        for inp in inputs:
            if isinstance(inp, str):
                normalized_inputs.append(EmbeddingInput(type="text", text=inp))
            elif isinstance(inp, dict):
                if inp.get("type") == "text":
                    normalized_inputs.append(EmbeddingInput(type="text", text=inp.get("text")))
                elif inp.get("type") == "image_url":
                    normalized_inputs.append(EmbeddingInput(type="image_url", image_url=inp.get("image_url")))
                else:
                    normalized_inputs.append(EmbeddingInput(**inp))
            else:
                normalized_inputs.append(inp)

        api_inputs = [inp.to_api_format() for inp in normalized_inputs]

        if use_cache:
            cached = _embedding_cache.get(self.model, api_inputs)
            if cached:
                return EmbeddingResponse(**cached)

        response = self._call_api(api_inputs)

        if use_cache:
            _embedding_cache.set(self.model, api_inputs, response.model_dump())

        return response

    def _call_api(self, inputs: list[dict]) -> EmbeddingResponse:
        """调用 API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
        }

        payload = {
            "model": self.model,
            "input": inputs,
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                self.url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        if isinstance(data["data"], dict):
            return EmbeddingResponse(
                object=data.get("object", "list"),
                data=[
                    EmbeddingResult(
                        index=0,
                        embedding=data["data"]["embedding"],
                        object="embedding",
                    )
                ],
                model=data.get("model", self.model),
                usage=data.get("usage", {}),
            )
        else:
            return EmbeddingResponse(
                object=data.get("object", "list"),
                data=[
                    EmbeddingResult(
                        index=item["index"] if isinstance(item, dict) and "index" in item else idx,
                        embedding=item["embedding"] if isinstance(item, dict) else item,
                        object=item.get("object", "embedding") if isinstance(item, dict) else "embedding",
                    )
                    for idx, item in enumerate(data["data"])
                ],
                model=data.get("model", self.model),
                usage=data.get("usage", {}),
            )

    def embed_text(self, text: str, use_cache: bool = True) -> list[float]:
        """嵌入单个文本

        Args:
            text: 文本内容
            use_cache: 是否使用缓存

        Returns:
            嵌入向量
        """
        response = self.embed([text], use_cache=use_cache)
        return response.data[0].embedding

    def embed_texts(self, texts: list[str], use_cache: bool = True) -> list[list[float]]:
        """嵌入多个文本

        Args:
            texts: 文本列表
            use_cache: 是否使用缓存

        Returns:
            嵌入向量列表
        """
        response = self.embed(texts, use_cache=use_cache)
        return [item.embedding for item in response.data]

    def embed_image(self, image_url: str, use_cache: bool = True) -> list[float]:
        """嵌入单个图像

        Args:
            image_url: 图像 URL
            use_cache: 是否使用缓存

        Returns:
            嵌入向量
        """
        response = self.embed(
            [EmbeddingInput(type="image_url", image_url={"url": image_url})],
            use_cache=use_cache,
        )
        return response.data[0].embedding

    def embed_multimodal(
        self,
        text: str | None = None,
        image_url: str | None = None,
        use_cache: bool = True,
    ) -> list[float]:
        """嵌入多模态内容（文本 + 图像）

        Args:
            text: 文本内容
            image_url: 图像 URL
            use_cache: 是否使用缓存

        Returns:
            嵌入向量
        """
        inputs = []
        if text:
            inputs.append(EmbeddingInput(type="text", text=text))
        if image_url:
            inputs.append(EmbeddingInput(type="image_url", image_url={"url": image_url}))

        if not inputs:
            raise ValueError("至少需要提供 text 或 image_url")

        response = self.embed(inputs, use_cache=use_cache)
        return response.data[0].embedding


# 默认的 Doubao 嵌入配置（当配置文件不可用时作为 fallback）
_DEFAULT_EMBEDDING = EmbeddingConfig(
    url="https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal",
    key="xxxx",
    model="doubao-embedding-vision-251215",
)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """计算两个向量的余弦相似度

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        余弦相似度，范围 [-1, 1]
    """
    if len(vec1) != len(vec2):
        raise ValueError("向量维度不一致")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
    """计算两个向量的欧几里得距离

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        欧几里得距离
    """
    if len(vec1) != len(vec2):
        raise ValueError("向量维度不一致")

    return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5
