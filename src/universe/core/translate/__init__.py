from .en import english
from ..config import Config

_config = Config()


def translate(text: str, lang: str | None = None) -> str:
    """将文本翻译为英文"""
    if lang == "zh":
        return text

    if lang is None:
        lang = _config.translate_lang

    try:
        translate_mapping = {
            'en': english,
        }[lang]
    except KeyError:
        return f"Unsupported language: {lang}"

    for key, value in translate_mapping.items():
        text = text.replace(key, value)

    return text


__all__ = [
    "translate",
]
