"""Pure-Logic Unit Tests for LLM Client components.

These tests cover isolated modules that benefit from deterministic testing:
1. ToolArgumentsValidator - type coercion and validation
2. LLMCache - filesystem-based caching with LRU eviction
"""

import pytest
import json
from pathlib import Path

from universe.core.llm_client.validator import ToolArgumentsValidator
from universe.core.llm_client.llm_cache import LLMCache


class TestToolArgumentsValidator:
    """Test ToolArgumentsValidator type coercion and validation logic"""

    @pytest.fixture
    def validator(self) -> ToolArgumentsValidator:
        return ToolArgumentsValidator()

    def test_string_to_array_coercion(self, validator: ToolArgumentsValidator):
        """String values should be coerced to arrays (parsed as JSON or wrapped)"""
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
            },
        }

        # Valid JSON array string
        arguments = {"items": '["a", "b", "c"]'}
        result = validator.validate(arguments, schema)
        assert result["items"] == ["a", "b", "c"]

        # Non-JSON string should be wrapped in array
        arguments = {"items": "single_item"}
        result = validator.validate(arguments, schema)
        assert result["items"] == ["single_item"]

    def test_string_to_number_coercion(self, validator: ToolArgumentsValidator):
        """String values should be coerced to numbers when expected"""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "value": {"type": "number"},
            },
        }

        arguments = {"count": "42", "value": "3.14"}
        result = validator.validate(arguments, schema)
        assert result["count"] == 42
        assert result["value"] == 3.14

    def test_boolean_coercion(self, validator: ToolArgumentsValidator):
        """String values should be coerced to booleans"""
        schema = {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
            },
        }

        # Various true-like strings
        for val in ["true", "True", "TRUE", "1", "yes", "YES"]:
            arguments = {"enabled": val}
            result = validator.validate(arguments, schema)
            assert result["enabled"] is True, f"Expected True for '{val}'"

        # Various false-like strings
        for val in ["false", "False", "FALSE", "0", "no", "NO", ""]:
            arguments = {"enabled": val}
            result = validator.validate(arguments, schema)
            assert result["enabled"] is False, f"Expected False for '{val}'"

    def test_required_field_validation(self, validator: ToolArgumentsValidator):
        """Missing required fields should raise ValueError"""
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "string"},
            },
            "required": ["required_field"],
        }

        # Missing required field
        with pytest.raises(ValueError, match="Missing required parameter"):
            validator.validate({}, schema)

        # Missing required field with optional present
        with pytest.raises(ValueError, match="Missing required parameter"):
            validator.validate({"optional_field": "value"}, schema)

        # All required fields present
        result = validator.validate({"required_field": "value"}, schema)
        assert result["required_field"] == "value"

    def test_string_to_object_coercion(self, validator: ToolArgumentsValidator):
        """String values should be coerced to objects (parsed as JSON)"""
        schema = {
            "type": "object",
            "properties": {
                "config": {"type": "object"},
            },
        }

        # Valid JSON object string
        arguments = {"config": '{"key": "value", "num": 123}'}
        result = validator.validate(arguments, schema)
        assert result["config"] == {"key": "value", "num": 123}

        # Invalid JSON should raise error
        arguments = {"config": "not a valid json"}
        with pytest.raises(ValueError, match="Cannot coerce string to object"):
            validator.validate(arguments, schema)

    def test_no_coercion_needed(self, validator: ToolArgumentsValidator):
        """Correctly typed values should pass through unchanged"""
        schema = {
            "type": "object",
            "properties": {
                "string_field": {"type": "string"},
                "int_field": {"type": "integer"},
                "bool_field": {"type": "boolean"},
                "array_field": {"type": "array"},
                "object_field": {"type": "object"},
            },
        }

        arguments = {
            "string_field": "hello",
            "int_field": 42,
            "bool_field": True,
            "array_field": [1, 2, 3],
            "object_field": {"a": 1},
        }

        result = validator.validate(arguments, schema)
        assert result == arguments


class TestLLMCache:
    """Test LLMCache filesystem-based caching with LRU eviction"""

    def test_cache_set_and_get(self, temp_cache_dir: Path):
        """Cache should store and retrieve values correctly"""
        cache = LLMCache(max_size=10, cache_dir=temp_cache_dir)

        messages = [{"role": "user", "content": "Hello"}]
        response = {"choices": [{"message": {"content": "Hi there!"}}]}

        # Initially should be a miss
        assert cache.get("gpt-4", messages, temperature=0.7) is None

        # Set the cache
        cache.set("gpt-4", messages, temperature=0.7, tools=None, response=response)

        # Should now be a hit
        cached_response = cache.get("gpt-4", messages, temperature=0.7)
        assert cached_response is not None
        assert cached_response["choices"][0]["message"]["content"] == "Hi there!"

    def test_cache_miss_different_params(self, temp_cache_dir: Path):
        """Different parameters should result in cache misses"""
        cache = LLMCache(max_size=10, cache_dir=temp_cache_dir)

        messages = [{"role": "user", "content": "Hello"}]
        response = {"choices": [{"message": {"content": "Hi there!"}}]}

        cache.set("gpt-4", messages, temperature=0.7, tools=None, response=response)

        # Same model, different temperature
        assert cache.get("gpt-4", messages, temperature=0.5) is None

        # Different model
        assert cache.get("gpt-3.5", messages, temperature=0.7) is None

        # Different messages
        different_messages = [{"role": "user", "content": "Goodbye"}]
        assert cache.get("gpt-4", different_messages, temperature=0.7) is None

    def test_cache_eviction_lru(self, temp_cache_dir: Path):
        """Cache should evict least recently used entries when full"""
        cache = LLMCache(max_size=3, cache_dir=temp_cache_dir)

        # Add 3 entries (fills the cache)
        for i in range(3):
            messages = [{"role": "user", "content": f"Message {i}"}]
            cache.set("gpt-4", messages, temperature=0.7, tools=None, response={"id": i})

        # Access first entry to make it recently used
        first_messages = [{"role": "user", "content": "Message 0"}]
        cache.get("gpt-4", first_messages, temperature=0.7)

        # Add 4th entry (should evict the second entry - least recently used)
        fourth_messages = [{"role": "user", "content": "Message 3"}]
        cache.set("gpt-4", fourth_messages, temperature=0.7, tools=None, response={"id": 3})

        # First entry should still be there (was accessed recently)
        assert cache.get("gpt-4", first_messages, temperature=0.7) is not None

        # Second entry should be evicted
        second_messages = [{"role": "user", "content": "Message 1"}]
        assert cache.get("gpt-4", second_messages, temperature=0.7) is None

        # Third entry should still be there
        third_messages = [{"role": "user", "content": "Message 2"}]
        assert cache.get("gpt-4", third_messages, temperature=0.7) is not None

    def test_cache_persistence(self, temp_cache_dir: Path):
        """Cache should persist to disk and be reloadable"""
        # Create first cache instance and add entry
        cache1 = LLMCache(max_size=10, cache_dir=temp_cache_dir)

        messages = [{"role": "user", "content": "Persistent message"}]
        response = {"choices": [{"message": {"content": "Persistent response"}}]}

        cache1.set("gpt-4", messages, temperature=0.7, tools=None, response=response)

        # Create second cache instance pointing to same directory
        cache2 = LLMCache(max_size=10, cache_dir=temp_cache_dir)

        # Should be able to retrieve the cached entry
        cached_response = cache2.get("gpt-4", messages, temperature=0.7)
        assert cached_response is not None
        assert cached_response["choices"][0]["message"]["content"] == "Persistent response"

    def test_cache_with_tools(self, temp_cache_dir: Path):
        """Cache should work correctly with tool definitions"""
        cache = LLMCache(max_size=10, cache_dir=temp_cache_dir)

        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
            }
        }]
        response = {"choices": [{"message": {"tool_calls": [{"function": {"name": "get_weather"}}]}}]}

        # Without tools - miss
        assert cache.get("gpt-4", messages, temperature=0.7, tools=None) is None

        # With tools - miss
        assert cache.get("gpt-4", messages, temperature=0.7, tools=tools) is None

        # Set with tools
        cache.set("gpt-4", messages, temperature=0.7, tools=tools, response=response)

        # With same tools - hit
        cached = cache.get("gpt-4", messages, temperature=0.7, tools=tools)
        assert cached is not None

        # Without tools - still miss (different key)
        assert cache.get("gpt-4", messages, temperature=0.7, tools=None) is None

    def test_cache_empty_messages(self, temp_cache_dir: Path):
        """Cache should handle empty messages list"""
        cache = LLMCache(max_size=10, cache_dir=temp_cache_dir)

        messages: list[dict[str, str]] = []
        response = {"choices": [{"message": {"content": "Empty request response"}}]}

        cache.set("gpt-4", messages, temperature=0.7, tools=None, response=response)

        cached = cache.get("gpt-4", messages, temperature=0.7)
        assert cached is not None
        assert cached["choices"][0]["message"]["content"] == "Empty request response"
