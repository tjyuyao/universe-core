from pathlib import Path


def find_project_root(start_path: Path) -> Path | None:
    """查找项目根目录，通过向上查找 pyproject.toml

    Args:
        start_path: 起始路径

    Returns:
        包含 pyproject.toml 的目录路径，如果未找到则返回 None
    """
    current = start_path.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def main() -> None:
    """主函数，移除所有 Python 文件行尾的空白字符，保留恰好一个末尾换行符"""

    current_file = Path(__file__)
    project_root = find_project_root(current_file)

    if project_root is None:
        print("❌ 未找到项目根目录（未找到 pyproject.toml）")
        return

    src_path = project_root / "src"

    if not src_path.exists():
        print(f"❌ src 目录不存在: {src_path}")
        return

    fixed_files: list[Path] = []

    for py_file in src_path.rglob("*.py"):
        if fix_file(py_file):
            fixed_files.append(py_file)

    print("-" * 50)
    if fixed_files:
        print(f"已修复 {len(fixed_files)} 个文件:")
        for f in fixed_files:
            print(f"  - {f.relative_to(project_root)}")
    else:
        print("✅ 所有文件都没有行尾空白字符")


def fix_file(file_path: Path) -> bool:
    """修复单个文件的行尾空白字符，保留恰好一个末尾换行符

    Args:
        file_path: Python 文件路径

    Returns:
        如果文件被修改则返回 True，否则返回 False
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"  ⚠️ 无法读取文件 {file_path}: {e}")
        return False

    # 处理空文件，避免误写入换行符
    if not content:
        return False

    # 统一处理换行符，使用 splitlines() 正确处理 \r\n 和 \n
    lines = content.splitlines()
    fixed_lines = [line.rstrip() for line in lines]

    # 使用 \n 连接各行，并保留恰好一个末尾换行符
    new_content = "\n".join(fixed_lines) + "\n"

    if new_content == content:
        return False

    try:
        file_path.write_text(new_content, encoding="utf-8", newline="")
        return True
    except OSError as e:
        print(f"  ⚠️ 无法写入文件 {file_path}: {e}")
        return False


if __name__ == "__main__":
    main()
