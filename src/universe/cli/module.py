"""Module subcommand implementations for universe CLI."""

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import tomllib

from universe.cli.registry import Registry, parse_module_spec


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
