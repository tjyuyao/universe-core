"""Module subcommand implementations for universe CLI."""

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import tomllib

from universe.cli.registry import Registry, parse_module_spec


def _generate_pyproject_toml(module_name: str, description: str) -> str:
    """Generate pyproject.toml content for a new module."""
    return f'''[project]
name = "universe-module-{module_name}"
version = "0.1.0"
description = "{description}"
requires-python = ">=3.14"
dependencies = [
    "universe>=0.1.0",
]

[tool.uv.sources]
universe = {{ git = "https://github.com/tjyuyao/universe" }}

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/universe_modules"]
'''


def _generate_init_py(module_name: str) -> str:
    """Generate __init__.py content for a new module."""
    return f'''"""Universe module: {module_name}.

Add your module's public API here.
"""

# Example:
# from .example_object import ExampleObject
# from .example_action import ExampleAction, ExampleParams
#
# __all__ = ["ExampleObject", "ExampleAction", "ExampleParams"]
'''


def _generate_example_object(module_name: str) -> str:
    """Generate example object file content."""
    class_name = "".join(word.capitalize() for word in module_name.split("_"))
    return f'''"""Example object for {module_name} module."""

from universe.core.object_ import Object, Action, Params, State, TimedStatus, ActionExecutionStatus


class {class_name}(Object):
    """An example object for the {module_name} module."""

    example_state: State[str]

    def __init__(self, object_id: str):
        super().__init__(object_id=object_id, actions=[ExampleAction()])
        self.example_state = "initial"


class ExampleAction(Action[{class_name}, Params]):
    """An example action."""

    name = "example_action"
    description = "An example action that does nothing."

    async def execute(self, obj, params, actor, world):
        obj.example_state = "modified"
        return TimedStatus(duration=0.5, status=ActionExecutionStatus.SUCCESS)
'''


def _generate_example_test(module_name: str) -> str:
    """Generate example test file content."""
    class_name = "".join(word.capitalize() for word in module_name.split("_"))
    return f'''"""Tests for {module_name} module."""

import pytest
from universe_modules.{module_name} import {class_name}


def test_{module_name}_creation():
    """Test that the object can be created."""
    obj = {class_name}("test_id")
    assert obj.object_id == "test_id"
    assert obj.example_state == "initial"
'''


def cmd_module_init(name: str, description: str | None) -> None:
    """Initialize a new universe module.

    Creates the standard module structure with namespace package layout.

    Examples:
        universe module init chat
        universe module init inventory --description "Items and containers"
    """
    # Validate module name (alphanumeric and underscores only)
    if not name.replace("_", "").isalnum():
        print(f"Error: Module name '{name}' is invalid.")
        print("Module names must contain only letters, numbers, and underscores.")
        sys.exit(1)

    # Check if directory already exists
    module_dir = Path(f"universe-module-{name}")
    if module_dir.exists():
        print(f"Error: Directory '{module_dir}' already exists.")
        sys.exit(1)

    # Use default description if not provided
    desc = description or f"Universe module: {name}"

    # Create directory structure
    src_dir = module_dir / "src" / "universe_modules" / name
    tests_dir = module_dir / "tests"

    src_dir.mkdir(parents=True)
    tests_dir.mkdir()

    # Create pyproject.toml
    pyproject_content = _generate_pyproject_toml(name, desc)
    (module_dir / "pyproject.toml").write_text(pyproject_content, encoding="utf-8")

    # Create __init__.py
    init_content = _generate_init_py(name)
    (src_dir / "__init__.py").write_text(init_content, encoding="utf-8")

    # Create example object file
    example_content = _generate_example_object(name)
    (src_dir / f"{name}_object.py").write_text(example_content, encoding="utf-8")

    # Create example test file
    test_content = _generate_example_test(name)
    (tests_dir / f"test_{name}.py").write_text(test_content, encoding="utf-8")

    # Create README.md
    readme_content = f"""# universe-module-{name}

{desc}

## Installation

```bash
universe module add {name}
```

## Usage

```python
from universe_modules.{name} import ExampleObject

obj = ExampleObject("my_object")
```

## Development

```bash
cd universe-module-{name}
uv sync
uv run pytest
```
"""
    (module_dir / "README.md").write_text(readme_content, encoding="utf-8")

    print(f"Created universe module '{name}' in '{module_dir}/'")
    print(f"")
    print(f"Structure:")
    print(f"  {module_dir}/")
    print(f"  ├── pyproject.toml")
    print(f"  ├── README.md")
    print(f"  ├── src/")
    print(f"  │   └── universe_modules/")
    print(f"  │       └── {name}/")
    print(f"  │           ├── __init__.py")
    print(f"  │           └── {name}_object.py")
    print(f"  └── tests/")
    print(f"      └── test_{name}.py")
    print(f"")
    print(f"Next steps:")
    print(f"  cd {module_dir}")
    print(f"  uv sync")
    print(f"  # Edit src/universe_modules/{name}/ to implement your module")


def handle_module_command(args: Namespace) -> None:
    """Dispatch module subcommands."""
    if args.module_command == "add":
        cmd_module_add(args.name)
    elif args.module_command == "remove":
        cmd_module_remove(args.name)
    elif args.module_command == "list":
        cmd_module_list()
    elif args.module_command == "search":
        cmd_module_search(args.query)
    elif args.module_command == "init":
        cmd_module_init(args.name, args.description)


def cmd_module_add(spec: str) -> None:
    """Add a universe module.

    Examples:
        universe module add chat
        universe module add spatial@v0.2.0
    """
    name, version = parse_module_spec(spec)
    registry = Registry.load()

    entry = registry.get(name)
    if entry is None:
        print(f"Error: Module '{name}' not found in registry.")
        print("Run 'universe module search' to see available modules.")
        sys.exit(1)

    pkg_name = f"universe-module-{name}"
    if version:
        pkg_spec = f"{pkg_name}>={version}"
        git_url = f"{entry.git}@{version}"
    else:
        pkg_spec = pkg_name
        git_url = entry.git

    cmd = ["uv", "add", pkg_spec, "--source", f"git+{git_url}"]
    print(f"Installing {pkg_spec}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install {pkg_spec}")
        sys.exit(e.returncode)


def cmd_module_remove(name: str) -> None:
    """Remove a universe module."""
    pkg_name = f"universe-module-{name}"
    cmd = ["uv", "remove", pkg_name]
    print(f"Removing {pkg_name}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to remove {pkg_name}")
        sys.exit(e.returncode)


def cmd_module_list() -> None:
    """List installed universe modules."""
    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        print("No pyproject.toml found in current directory.")
        return

    content = pyproject.read_text(encoding="utf-8")
    config = tomllib.loads(content)

    deps = config.get("project", {}).get("dependencies", [])
    universe_modules = [d for d in deps if d.startswith("universe-module-")]

    if not universe_modules:
        print("No universe modules installed.")
        return

    print("Installed universe modules:")
    for mod in universe_modules:
        print(f"  - {mod}")


def cmd_module_search(query: str | None) -> None:
    """Search available modules in registry."""
    registry = Registry.load()

    entries = registry.search(query)
    if not entries:
        if query:
            print(f"No modules found matching '{query}'.")
        else:
            print("No modules available in registry.")
        return

    print("Available universe modules:")
    for name, entry in entries:
        print(f"  - {name}: {entry.description}")
        print(f"    {entry.git}")
