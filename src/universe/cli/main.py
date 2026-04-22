"""Main entry point and argument parsing for universe CLI."""

import argparse

from universe.cli.module import handle_module_command


def run_cli() -> None:
    """Run the universe CLI."""
    parser = argparse.ArgumentParser(
        prog="universe",
        description="Universe Framework CLI - Package manager for universe modules"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # module subcommand
    module_parser = subparsers.add_parser("module", help="Manage universe modules")
    module_subparsers = module_parser.add_subparsers(
        dest="module_command",
        required=True
    )

    # module add
    add_parser = module_subparsers.add_parser("add", help="Install a universe module")
    add_parser.add_argument(
        "name",
        help="Module name (e.g., 'chat' or 'chat@v0.2.0')"
    )

    # module remove
    remove_parser = module_subparsers.add_parser(
        "remove",
        help="Uninstall a universe module"
    )
    remove_parser.add_argument("name", help="Module name")

    # module list
    module_subparsers.add_parser("list", help="List installed universe modules")

    # module search
    search_parser = module_subparsers.add_parser(
        "search",
        help="Search available modules"
    )
    search_parser.add_argument(
        "query",
        nargs="?",
        help="Search query (optional)"
    )

    # module init
    init_parser = module_subparsers.add_parser(
        "init",
        help="Initialize a new universe module"
    )
    init_parser.add_argument(
        "name",
        help="Module name (e.g., 'chat', 'inventory')"
    )
    init_parser.add_argument(
        "--description",
        help="Module description",
        default=None
    )

    args = parser.parse_args()

    if args.command == "module":
        handle_module_command(args)
