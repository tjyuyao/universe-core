import ast
from pathlib import Path


def main() -> None:
    """主函数，检查所有依赖顺序"""

    all_orders: dict[str, DependencyOrder] = {}

    all_orders["core"] = DependencyOrder("core", [
        "meta",
        "config",
        "translate",
        "timing",
        "llm_client",
        "object_",
        "memory",
        "agent",
        "universe",
        ], exclude_dirs=["linter"])

    all_orders["llm_client"] = DependencyOrder("core/llm_client", [
        "budget",
        "validator",
        "llm_cache",
        "llm_logger",
        "llm_client",
        "__init__",
        ])

    all_orders["object_"] = DependencyOrder("core/object_", [
        "state",
        "serializable",
        "appearance",
        "object_",
        "__init__",
        ])

    all_orders["agent"] = DependencyOrder("core/agent", [
        "mindset",
        "role",
        "soul",
        "attention",
        "agent",
        "__init__",
        ])

    all_orders["universe"] = DependencyOrder("core/universe", [
        "world",
        "universe",
        "__init__",
        ])

    all_passed = True
    for order in all_orders.values():
        if not order.check_and_report():
            all_passed = False

    print("\n" + "-" * 50)
    if all_passed:
        print("总结：✅ 所有依赖检查通过")
    else:
        print("总结：❌ 存在依赖违规")


class DependencyOrder:
    """依赖顺序：排序在后的模块不可从排序较前的模块导入，不能有模块未被列出。"""

    def __init__(self, path: str, order: list[str], exclude_dirs: list[str] | None = None):
        self.path = path
        self.order = order
        self.exclude_dirs = set(exclude_dirs or [])

    def __str__(self) -> str:
        return self.path + ":" + " -> ".join(self.order)

    def _get_base_path(self) -> Path:
        """获取需要检查的目录路径"""
        current_file = Path(__file__)
        core_path = current_file.parent.parent
        # 处理 "core" -> core_path, "core/xxx" -> core_path/xxx
        if self.path == "core":
            return core_path
        return core_path / self.path.replace("core/", "")

    def _get_module_name(self, file_path: Path) -> str | None:
        """从文件路径推断模块名

        规则：
        - core/singleton.py -> "singleton"
        - core/llm_client/llm_cache.py -> "llm_client" (父目录名)
        - core/llm_client/__init__.py -> "llm_client" (package 名)
        """
        base_path = self._get_base_path()
        rel_path = file_path.relative_to(base_path)
        parts = list(rel_path.parts)

        # 移除 .py 后缀
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]

        # 如果是子目录中的文件
        if len(parts) >= 2:
            # 返回父目录名作为模块名
            return parts[-2]
        else:
            # 直接子文件
            return parts[-1]

    def _parse_imports(self, file_path: Path) -> list[tuple[str, int]]:
        """解析 Python 文件中的所有导入模块，返回 (模块名, 行号)"""
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            return []

        # 获取 TYPE_CHECKING 块的行号范围
        type_checking_ranges = self._get_type_checking_ranges(tree)

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # 跳过 TYPE_CHECKING 块内的导入
                if self._is_in_type_checking_block(node.lineno, type_checking_ranges):
                    continue
                for alias in node.names:
                    imports.append((alias.name.split(".")[0], node.lineno))
            elif isinstance(node, ast.ImportFrom):
                # 跳过 TYPE_CHECKING 块内的导入
                if self._is_in_type_checking_block(node.lineno, type_checking_ranges):
                    continue
                if node.module:
                    # 绝对导入: from module import xxx
                    imports.append((node.module.split(".")[0], node.lineno))
                elif node.level > 0:
                    # 相对导入: from .xxx, from ..xxx
                    current_module = self._get_module_name(file_path)
                    if current_module and current_module in self.order:
                        # 解析相对导入的模块
                        imported = self._resolve_relative_import(
                            file_path, node.level, node.names[0].name if node.names else ""
                        )
                        if imported:
                            imports.append((imported, node.lineno))
        return imports

    def _get_type_checking_ranges(self, tree: ast.AST) -> list[tuple[int, int]]:
        """获取所有 TYPE_CHECKING 块的行号范围 [(start, end), ...]"""
        ranges = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # 检查是否是 TYPE_CHECKING 判断
                if self._is_type_checking_check(node.test):
                    # 获取该 if 块的起始和结束行号
                    start = node.lineno
                    end = self._get_last_line(node)
                    ranges.append((start, end))
        return ranges

    def _is_type_checking_check(self, node: ast.AST) -> bool:
        """检查节点是否是 TYPE_CHECKING 的判断"""
        # 直接: if TYPE_CHECKING
        if isinstance(node, ast.Name) and node.id == "TYPE_CHECKING":
            return True
        # 检查 from typing import TYPE_CHECKING 后的使用
        # if typing.TYPE_CHECKING
        if isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING":
            return True
        return False

    def _get_last_line(self, node: ast.If) -> int:
        """获取 AST 节点的最后一行行号"""
        last_line = node.lineno
        for child in ast.walk(node):
            if hasattr(child, 'lineno'):
                lineno = getattr(child, 'lineno')
                if isinstance(lineno, int) and lineno > last_line:
                    last_line = lineno
        return last_line

    def _is_in_type_checking_block(self, lineno: int, ranges: list[tuple[int, int]]) -> bool:
        """检查行号是否在 TYPE_CHECKING 块内"""
        for start, end in ranges:
            if start <= lineno <= end:
                return True
        return False

    def _resolve_relative_import(
        self, file_path: Path, level: int, name: str
    ) -> str | None:
        """解析相对导入为模块名"""
        current_dir = file_path.parent

        # 向上回溯 level 层
        for _ in range(level - 1):
            current_dir = current_dir.parent

        # 如果有名字，可能是子模块
        if name:
            # 尝试在当前目录下找子目录
            potential_module = current_dir / name
            if potential_module.is_dir() and (potential_module / "__init__.py").exists():
                return name
            # 否则返回目录名
            return current_dir.name
        else:
            return current_dir.name

    def check_and_report(self) -> bool:
        """检查依赖顺序是否符合要求

        规则：排序在后的模块不可从排序较前的模块导入
        """
        module_to_index = {name: i for i, name in enumerate(self.order)}
        violations: list[dict] = []
        unlisted_modules = set()

        base_path = self._get_base_path()

        if not base_path.exists():
            print(f"路径不存在: {base_path}")
            return False

        # 遍历所有 Python 文件
        for py_file in base_path.rglob("*.py"):
            # 跳过排除的目录
            if any(excluded in py_file.parts for excluded in self.exclude_dirs):
                continue

            current_module = self._get_module_name(py_file)

            if current_module is None:
                continue

            if current_module not in module_to_index:
                unlisted_modules.add(current_module)
                continue

            current_index = module_to_index[current_module]

            # 解析该文件的所有导入
            imported_modules = self._parse_imports(py_file)

            for imported, lineno in imported_modules:
                if imported not in module_to_index:
                    continue

                imported_index = module_to_index[imported]

                # 检查违规：导入的模块索引 > 当前模块索引 (即排序在后的模块)
                if imported_index > current_index:
                    violations.append(
                        {
                            "file": str(py_file.relative_to(base_path.parent.parent)),
                            "from": current_module,
                            "to": imported,
                            "file_index": current_index,
                            "import_index": imported_index,
                            "line": lineno,
                        }
                    )

        # 报告结果
        has_error = False

        if unlisted_modules:
            print(f"\n❌ [{self.path}] 发现未列出的模块:")
            for mod in sorted(unlisted_modules):
                print(f"   - {mod}")
            has_error = True

        if violations:
            print(f"\n❌ [{self.path}] 发现循环依赖违规 ({len(violations)} 处):")
            # 按文件分组
            by_file: dict[str, list] = {}
            for v in violations:
                file_key: str = v["file"]
                assert isinstance(file_key, str)
                if file_key not in by_file:
                    by_file[file_key] = []
                by_file[file_key].append(v)

            for file, vlist in sorted(by_file.items()):
                print(f"\n   {file}")
                # 读取文件内容以显示代码
                file_path = base_path.parent.parent / file
                try:
                    lines = file_path.read_text(encoding="utf-8").splitlines()
                except:
                    lines = []
                for v in vlist:
                    line_no = v['line']
                    code_line = lines[line_no - 1].strip() if lines and line_no <= len(lines) else ""
                    print(f"      [L{line_no}] {code_line}")
            has_error = True
            print("")

        if not has_error:
            print(f"✅ [{self.path}] 依赖检查通过")
            return True

        return False


if __name__ == "__main__":
    main()
