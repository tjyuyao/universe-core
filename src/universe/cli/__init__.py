"""Universe CLI - Package manager for universe modules."""

from universe.cli.registry import Registry, RegistryEntry, parse_module_spec
from universe.cli.module import (
    handle_module_command,
    cmd_module_add,
    cmd_module_remove,
    cmd_module_list,
    cmd_module_search,
)

__all__ = [
    "Registry",
    "RegistryEntry",
    "parse_module_spec",
    "handle_module_command",
    "cmd_module_add",
    "cmd_module_remove",
    "cmd_module_list",
    "cmd_module_search",
]
