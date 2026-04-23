# Universe CLI (Package Manager)

## Overview

Universe modules are reusable building blocks that extend the core framework. Each module is an independent Python package that depends on `universe-core` (the core) and can be shared across projects and domains.

The package management strategy has three layers:

```
universe module add chat          ← thin CLI (alias layer)
    ↓
uv add universe-module-chat ...   ← uv (dependency resolution, locking, install)
    ↓
git+https://github.com/.../...    ← git (source of truth)
```

- **Module packaging convention** defines how to structure a module.
- **uv** handles dependency resolution, version locking, and installation.
- **`universe module` CLI** is a thin alias layer that maps short module names to git URLs via a registry.

## Architecture

```
universe-core/              (framework kernel — Object, Agent, World)
universe-module-chat/       (reusable module — Inbox, SendMessageAction, ...)
universe-module-spatial/    (reusable module — Room, Door, MoveAction, ...)
app-game/                   (application — depends on universe + chat + spatial)
app-research/               (application — depends on universe + chat)
```

Rules:
- Modules depend on `universe-core`. Never the reverse.
- Modules do NOT depend on other modules by default (flat strategy).
- If a module must depend on another module, declare it explicitly in its `pyproject.toml`.

## Module Structure

A module is a minimal Python package following the `src/` layout with an implicit namespace package:

```
universe-module-chat/
    pyproject.toml
    src/
        universe_modules/           ← namespace package (NO __init__.py!)
            chat/
                __init__.py         ← module public API
                inbox.py            ← Object subclass
                actions.py          ← Action definitions
    tests/
        test_chat.py
```

**Critical**: the `universe_modules/` directory must NOT contain `__init__.py`. It is a Python [implicit namespace package](https://peps.python.org/pep-0420/) that allows multiple independently-installed modules to coexist under `universe_modules.*`.

**Note**: you can use `universe module init` to automatically create a new module.

```bash
universe module init chat
```

This creates a new module `universe-module-chat` with a basic structure.

### pyproject.toml

```toml
[project]
name = "universe-module-chat"
version = "0.1.0"
description = "Chat and messaging objects for Universe"
requires-python = ">=3.14"
dependencies = [
    "universe-core>=0.1.0",
]

[tool.uv.sources]
universe-core = { git = "https://github.com/tjyuyao/universe-core" }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/universe_modules"]
```

### __init__.py

Export the module's public API:

```python
from .inbox import Inbox
from .actions import SendMessageAction, SendMessageParams

__all__ = ["Inbox", "SendMessageAction", "SendMessageParams"]
```

### Module code

Modules only import from `universe.core`:

```python
from universe.core.object_ import Object, Action, Params, State, Channel, TimedStatus, ActionExecutionStatus
from universe.core.agent import Agent
from universe.core.universe import World

class Inbox(Object):
    messages: State[list[str]]

    def __init__(self, object_id: str):
        super().__init__(object_id=object_id, actions=[SendMessageAction()])
        self.messages = []
```

## Using Modules in Applications

### Install

```bash
universe module add chat
universe module add spatial@v0.2.0    # pinned version
```

This resolves to `uv add` with the correct git source.

### Import

```python
from universe.core.object_ import Object, State
from universe.core.agent import Agent
from universe.core.universe import World

from universe_modules.chat import Inbox, SendMessageAction
from universe_modules.spatial import Room, Door, MoveAction

world = World(name="game", description="A text adventure")
world.register_object("lobby", Room("lobby"))
world.register_object("inbox", Inbox("inbox"))
```

### Resulting pyproject.toml

After `universe module add chat spatial`, the application's `pyproject.toml` looks like:

```toml
[project]
name = "my-game"
dependencies = [
    "universe-core>=0.1.0",
    "universe-module-chat>=0.1.0",
    "universe-module-spatial>=0.2.0",
]

[tool.uv.sources]
universe-core = { git = "https://github.com/tjyuyao/universe-core" }
universe-module-chat = { git = "https://github.com/tjyuyao/universe-module-chat" }
universe-module-spatial = { git = "https://github.com/tjyuyao/universe-module-spatial", tag = "v0.2.0" }
```

### Remove

```bash
universe module remove chat
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `universe module add <name>[@<version>]` | Install a module (resolves via registry, delegates to `uv add`) |
| `universe module remove <name>` | Uninstall a module (delegates to `uv remove`) |
| `universe module list` | List installed universe modules |
| `universe module search [<query>]` | Search the registry for available modules |

### How the CLI works

The CLI is a thin wrapper (~50 lines). It does NOT implement dependency resolution — that is uv's job. It only:

1. Reads the **registry** to resolve a short name (`chat`) to a git URL.
2. Translates to the corresponding `uv` command.
3. Runs it.

```
universe module add chat@v0.2.0
    ↓ registry lookup
uv add "universe-module-chat>=0.2.0" --source "git+https://github.com/tjyuyao/universe-module-chat@v0.2.0"
```

## Registry

The registry is a TOML file that maps module short names to their git URLs. It is bundled with `universe-core` at `src/registry.toml`:

```toml
# Universe Module Registry
# Maps short names to git repositories.

[chat]
git = "https://github.com/tjyuyao/universe-module-chat"
description = "Chat and messaging: Inbox, SendMessageAction"

[spatial]
git = "https://github.com/tjyuyao/universe-module-spatial"
description = "Spatial world: Room, Door, MoveAction"

[inventory]
git = "https://github.com/tjyuyao/universe-module-inventory"
description = "Items and containers: Inventory, Item, TradeAction"
```

To add a new module to the ecosystem, add an entry to this file in `universe-core`.

## Versioning

- Modules use **git tags** for versioning: `v0.1.0`, `v0.2.0`, etc.
- `universe module add chat` installs the latest (no tag constraint).
- `universe module add chat@v0.2.0` pins to a specific tag.
- The `uv.lock` file in each application pins the exact commit, ensuring reproducible builds.

## Creating a New Module (Checklist)

1. Create a new repository named `universe-module-<name>`.
2. Set up the directory structure (see [Module Structure](#module-structure)).
3. Ensure `src/universe_modules/` has **no** `__init__.py`.
4. Ensure all imports come from `universe.core`, not from other modules.
5. Write tests following the e2e pattern: build a World, register objects, run `world.step()`.
6. Tag a release: `git tag v0.1.0 && git push --tags`.
7. Add an entry to `universe/src/registry.toml`.
