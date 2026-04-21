"""Tool Arguments Validator - 验证和修复 LLM 返回的 tool_call 参数

根据 JSON Schema 定义，验证参数类型并自动修复常见错误。
"""

from typing import Any


class ToolArgumentsValidator:
    """验证和修复 LLM 返回的 tool_call 参数

    根据 JSON Schema 定义，验证参数类型并自动修复常见错误。

    Example:
        >>> validator = ToolArgumentsValidator()
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "content": {"type": "string"},
        ...         "mentions": {"type": "array", "items": {"type": "string"}}
        ...     },
        ...     "required": ["content"]
        ... }
        >>> arguments = {"content": "hello", "mentions": "[]"}
        >>> validator.validate(arguments, schema)
        {"content": "hello", "mentions": []}
    """

    def validate(self, arguments: dict, schema: dict) -> dict:
        """验证并修复参数

        Args:
            arguments: LLM 返回的参数
            schema: JSON Schema 定义

        Returns:
            修复后的参数

        Raises:
            ValueError: 缺少必填字段或无法修复的类型错误
        """
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for field_name, field_schema in properties.items():
            if field_name not in arguments:
                if field_name in required:
                    raise ValueError(f"Missing required parameter: {field_name}")
                continue

            value = arguments[field_name]
            expected_type = field_schema.get("type")

            if expected_type and not self._is_type_match(value, expected_type):
                arguments[field_name] = self._coerce_value(value, expected_type, field_schema)

        return arguments

    def _is_type_match(self, value: Any, expected_type: str) -> bool:
        """检查值是否符合期望的 JSON Schema 类型

        Args:
            value: 实际值
            expected_type: JSON Schema 类型（string, number, integer, boolean, array, object, null）

        Returns:
            是否匹配
        """
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        elif expected_type == "null":
            return value is None
        return True

    def _coerce_value(self, value: Any, expected_type: str, field_schema: dict) -> Any:
        """将值转换为期望的类型

        Args:
            value: 实际值
            expected_type: JSON Schema 类型
            field_schema: 字段 schema 定义

        Returns:
            转换后的值

        Raises:
            ValueError: 无法转换时抛出
        """
        import json

        if expected_type == "array" and isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return [value]

        elif expected_type == "object" and isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Cannot coerce string to object: {value}")

        elif expected_type == "string" and not isinstance(value, str):
            return str(value)

        elif expected_type == "number" and isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Cannot coerce string to number: {value}")

        elif expected_type == "integer" and isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"Cannot coerce string to integer: {value}")

        elif expected_type == "boolean" and isinstance(value, str):
            return value.lower() in ("true", "1", "yes")

        return value
