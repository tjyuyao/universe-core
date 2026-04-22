"""Registry parsing and lookup for universe modules."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast


@dataclass(frozen=True)
class RegistryEntry:
    """A module entry in the registry."""

    git: str
    description: str


class Registry:
    """Module registry - maps short names to git URLs."""

    _instance: Registry | None = None

    def __init__(self, entries: dict[str, RegistryEntry]) -> None:
        self._entries = entries

    @classmethod
    def load(cls) -> Self:
        """Load registry from bundled registry.toml."""
        if cls._instance is None:
            registry_path = Path(__file__).parent.parent.parent / "registry.toml"
            content = registry_path.read_text(encoding="utf-8")
            data = tomllib.loads(content)

            entries: dict[str, RegistryEntry] = {}
            for name, info in data.items():
                if isinstance(info, dict):
                    entries[name] = RegistryEntry(
                        git=info.get("git", ""),
                        description=info.get("description", "")
                    )

            cls._instance = cls(entries)
        return cast(Self, cls._instance)

    def get(self, name: str) -> RegistryEntry | None:
        """Get registry entry by module name."""
        return self._entries.get(name)

    def search(self, query: str | None) -> list[tuple[str, RegistryEntry]]:
        """Search registry by query string."""
        if query is None:
            return list(self._entries.items())

        query = query.lower()
        results: list[tuple[str, RegistryEntry]] = []
        for name, entry in self._entries.items():
            if query in name.lower() or query in entry.description.lower():
                results.append((name, entry))
        return results


def parse_module_spec(spec: str) -> tuple[str, str | None]:
    """Parse 'name' or 'name@version' into (name, version)."""
    if "@" in spec:
        name, version = spec.split("@", 1)
        return name, version
    return spec, None
