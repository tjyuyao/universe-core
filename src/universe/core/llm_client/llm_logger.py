import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


from ..config import LLMConfig


class LLMLogger():
    """LLM 日志记录器"""
    
    def __init__(self, request_log_dir: Path, max_log_sessions: int = 10) -> None:
        self._request_log_dir = request_log_dir
        self._max_log_sessions = max_log_sessions
        self._current_session_dir = self._request_log_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self._current_session_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_old_sessions()

    def write_request_log(
        self,
        timestamp: datetime,
        config: LLMConfig,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None,
        response_data: dict[str, Any],
        cache_hit: bool,
    ) -> Path:
        """写入请求日志到 markdown 文件

        Args:
            timestamp: 请求时间戳
            config: LLM 配置
            messages: 请求消息列表
            tools: 工具定义列表
            response_data: API 响应数据
            cache_hit: 是否缓存命中

        Returns:
            日志文件路径
        """
        # 生成文件名：timestamp.md，保存在当前 session 文件夹中
        filename = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".md"
        log_file = self._current_session_dir / filename

        # 提取响应信息
        choices = response_data.get("choices", [{}])
        message = choices[0].get("message", {}) if choices else {}
        response_content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls", [])

        # 构建消息部分（markdown 格式）
        messages_md = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            messages_md.append(f"**[{i+1}] {role}**\n\n{content}\n")

        # 构建工具部分（markdown 格式）
        tools_md = []
        if tools:
            for i, tool in enumerate(tools):
                tool_type = tool.get("type", "unknown")
                func = tool.get("function", {})
                func_name = func.get("name", "unknown")
                func_desc = func.get("description", "")
                tools_md.append(f"**[{i+1}] {func_name}** (`{tool_type}`)\n\n{func_desc}\n")

        # 构建工具调用部分（markdown 格式）
        tool_calls_md = []
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                func = tc.get("function", {})
                func_name = func.get("name", "unknown")
                func_args = func.get("arguments", "")
                tool_calls_md.append(f"**[{i+1}] {func_name}**\n\nArguments: `{func_args}`\n")

        # 构建 markdown 内容
        md_lines = [
            "# LLM Request Log",
            "",
            "## Request",
            f"- **Timestamp**: {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}",
            f"- **Model**: {config.model}",
            f"- **Messages**: {len(messages)}",
            f"- **Tools**: {len(tools) if tools else 0}",
            f"- **Cache Hit**: {'true' if cache_hit else 'false'}",
            "",
            "### Messages",
            "",
            *messages_md,
        ]

        if tools:
            md_lines.extend([
                "### Tools",
                "",
                *tools_md,
            ])

        md_lines.extend([
            "## Response",
            f"- **Content Length**: {len(response_content)}",
            f"- **Tool Calls**: {len(tool_calls)}",
            "",
            "### Content",
            "",
            response_content if response_content else "*(no content)*",
            "",
        ])

        if tool_calls:
            md_lines.extend([
                "### Tool Calls",
                "",
                *tool_calls_md,
            ])

        # 写入文件
        log_file.write_text("\n".join(md_lines), encoding="utf-8")

        return log_file

    def _cleanup_old_sessions(self) -> None:
        """清理旧的 session 文件夹，只保留最新的 _max_log_sessions 个"""
        # 获取所有 session 文件夹（直接子目录）
        session_dirs = [d for d in self._request_log_dir.iterdir() if d.is_dir()]

        if len(session_dirs) <= self._max_log_sessions:
            return

        # 按修改时间排序（最旧的在前）
        session_dirs.sort(key=lambda d: d.stat().st_mtime)

        # 删除最旧的文件夹
        dirs_to_delete = len(session_dirs) - self._max_log_sessions
        for old_dir in session_dirs[:dirs_to_delete]:
            shutil.rmtree(old_dir, ignore_errors=True)