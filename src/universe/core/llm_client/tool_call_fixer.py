"""Tool call 格式修复器

处理各种模型可能返回的异常工具调用格式。
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..object_ import Channel
    from ..universe.world import World


def _match_args_to_schema(
    args_dict: dict,
    required_params: list[str],
    action_name: str,
) -> dict | None:
    """将参数字典匹配到 schema 的 required 参数

    Args:
        args_dict: 候选参数字典（不含空键）
        required_params: schema 中 required 的参数名列表
        action_name: 用于警告信息的 action 名

    Returns:
        匹配后的参数字典，如果无法匹配则返回 None
    """
    # 情况1: 参数键完全匹配 required 参数
    args_keys = set(args_dict.keys())
    required_set = set(required_params)

    if args_keys == required_set:
        return args_dict

    # 情况2: 单参数函数，只有一个值需要包装
    if len(required_params) == 1 and len(args_dict) == 1:
        param_name = required_params[0]
        value = list(args_dict.values())[0]
        return {param_name: value}

    # 情况3: 参数是子集（有些参数有默认值）
    if args_keys <= required_set:
        return args_dict

    # 无法匹配
    return None


def recover_action_from_tool_call(
    arguments: dict,
    action_name: str,
    all_channels: dict[str, Channel],
    world: World,
) -> tuple[str | None, dict]:
    """从异常 tool call 中恢复 action name 并修复参数结构

    处理以下异常格式：
    1. {"i_have_reasoned_that": "reasoning content"} (name 在 arguments 的 key 中)
        修复逻辑：遍历 arguments 的 keys，找到匹配的 action name，用正确的参数名包装 value
    2. {"": "i_have_reasoned_that", "content": "value"} (空键值对格式)
        修复逻辑：从空键获取 action name，根据 schema 匹配剩余参数

    Args:
        arguments: 工具调用的参数字典
        action_name: 当前的工具调用名称（可能为空）
        all_channels: 所有可用的 Channel
        world: World 实例

    Returns:
        (action_name, fixed_arguments): 修复后的 action name 和参数
        如果无法修复，返回 (None, arguments)
    """
    from ..object_ import Action

    # 首先尝试新格式：空字符串键包含 action name
    if "" in arguments:
        potential_name = arguments[""]
        if potential_name and isinstance(potential_name, str):
            # 查找支持该 action name 的 channel
            matching_channels = [
                ch for ch in all_channels.values()
                if ch.has_action(potential_name, world)
            ]
            if matching_channels:
                # 获取第一个匹配的 action 来确定参数结构
                channel = matching_channels[0]
                action = channel.get_action(potential_name, world)
                params_type = action.GetParamsType()
                schema = params_type.model_json_schema()
                required = schema.get("required", [])

                # 获取非空键的参数
                candidate_args = {k: v for k, v in arguments.items() if k != ""}

                # 尝试根据 schema 匹配参数
                fixed_args = _match_args_to_schema(
                    candidate_args, required, potential_name
                )

                if fixed_args is not None:
                    warnings.warn(
                        f"Recovered action '{potential_name}' from empty key, "
                        f"mapped to params {list(fixed_args.keys())}"
                    )
                    return potential_name, fixed_args

                # 如果候选参数为空但 action 不需要参数
                if not candidate_args and not required:
                    warnings.warn(
                        f"Recovered action '{potential_name}' from empty key (no params required)"
                    )
                    return potential_name, {}

    # 然后尝试旧格式：action name 在 arguments 的 key 中
    for potential_name in list(arguments.keys()):
        if potential_name == "":
            continue  # 已经处理过

        # 查找支持该 action name 的 channel
        matching_channels = [
            ch for ch in all_channels.values()
            if ch.has_action(potential_name, world)
        ]
        if not matching_channels:
            continue

        # 获取第一个匹配的 action
        channel = matching_channels[0]
        action = channel.get_action(potential_name, world)

        # 从 action 的 Params 类型获取参数 schema
        params_type = action.GetParamsType()
        schema = params_type.model_json_schema()
        required = schema.get("required", [])

        # 获取嵌套的 value
        nested_value = arguments[potential_name]

        # 如果已经是 dict 格式，直接使用
        if isinstance(nested_value, dict):
            return potential_name, nested_value

        # 单参数函数：用第一个 required 参数名包装 value
        if len(required) == 1 and nested_value is not None:
            param_name = required[0]
            fixed_args = {param_name: nested_value}
            warnings.warn(
                f"Recovered action '{potential_name}' with param '{param_name}'"
            )
            return potential_name, fixed_args

        # 无法自动修复，跳过
        warnings.warn(
            f"Cannot auto-fix action '{potential_name}': "
            f"requires {len(required)} params, got non-dict value"
        )

    return None, arguments


def fix_tool_call_format(
    tool_call: dict,
    all_channels: dict[str, Channel],
    world: World,
) -> dict:
    """修复单个 tool call 的格式

    Args:
        tool_call: 包含 name 和 arguments 的工具调用字典
        all_channels: 所有可用的 Channel
        world: World 实例

    Returns:
        修复后的 tool_call 字典
    """
    arguments = tool_call.get("arguments", {})
    if isinstance(arguments, str):
        import json
        arguments = json.loads(arguments)

    action_name = tool_call.get("name", "")

    # 如果 name 为空，尝试恢复
    if not action_name:
        recovered_name, fixed_args = recover_action_from_tool_call(
            arguments, action_name, all_channels, world
        )
        if recovered_name:
            tool_call["name"] = recovered_name
            tool_call["arguments"] = fixed_args

    return tool_call
