# Object Subclassing Quickstart

## Overview

`Object` is the base class for all entities in the simulation world. Objects have:

- **State** (`State[T]` annotations) -- serializable fields that represent the object's observable state
- **Actions** -- operations that Agents can invoke on the object via LLM tool calls
- **Activities** -- a queue of pending action requests, processed in logical time order

An `Object` is passive by default: it exposes state for observation and receives actions from Agents.

## Minimal Example

```python
from universe.core.object_ import Object, Action, Params, TimedStatus, ActionExecutionStatus

class Light(Object):
    """A light that can be turned on or off."""

    is_on: State[bool]

    def __init__(self, object_id: str):
        super().__init__(
            object_id=object_id,
            actions=[ToggleAction()],
            read_speed=20,  # tokens/second when observed
        )
        self.is_on = False


class ToggleAction(Action[Light, Params]):
    name = "toggle_light"
    description = "Toggle the light on or off."

    async def execute(self, obj, params, actor, world):
        obj.is_on = not obj.is_on
        return TimedStatus(duration=0.5, status=ActionExecutionStatus.SUCCESS)
```

## Defining State

Use `State[T]` type annotations to mark fields as serializable state. These fields are automatically included in `state_dict()` (used by `observe()` to build the LLM context).

```python
from universe.core.object_ import Object, State

class Door(Object):
    is_locked: State[bool]
    durability: State[float]
    label: State[str]

    def __init__(self, object_id: str):
        super().__init__(object_id=object_id)
        self.is_locked = True
        self.durability = 100.0
        self.label = "Front Door"
```

Supported state types: `int`, `float`, `str`, `bool`, `None`, `list`, `tuple`, `dict`, and Pydantic `BaseModel` subclasses. All state values must be JSON-serializable.

Fields **not** annotated with `State[T]` (e.g. plain `dict`, local caches) are invisible to serialization and observation.

## Defining Actions

Actions are how Agents interact with Objects. Each `Action` is generic over the Object type and a `Params` model:

```python
from pydantic import BaseModel
from universe.core.object_ import (
    Action, Params, TimedStatus, ActionExecutionStatus, State,
)


# 1. Define parameters (use Params base for no-arg actions)
class UnlockParams(BaseModel):
    key_id: str


# 2. Define the action
class UnlockAction(Action["Door", UnlockParams]):
    name = "unlock"
    description = "Unlock the door with a key."

    async def execute(self, obj, params, actor, world):
        if params.key_id == "master":
            obj.is_locked = False
            return TimedStatus(duration=2.0, status=ActionExecutionStatus.SUCCESS)
        return TimedStatus(duration=1.0, status=ActionExecutionStatus.FAIL)
```

Key points:

- `name` becomes the LLM tool name. Must be unique within the Object.
- `description` is passed to the LLM as the tool description.
- The `Params` model's JSON schema is automatically used as the tool's `parameters` schema.
- `execute()` must return a `TimedStatus` with:
  - `duration` -- how much logical time the action consumes.
  - `status` -- `SUCCESS` or `FAIL`.

Register actions in `__init__`:

```python
class Door(Object):
    is_locked: State[bool]

    def __init__(self, object_id: str):
        super().__init__(
            object_id=object_id,
            actions=[UnlockAction(), LockAction()],
        )
        self.is_locked = True
```

## Overriding `observe()`

By default, `observe()` serializes the entire `state_dict()` as HJSON. Override it to control what Agents perceive:

```python
from universe.core.timing import TimedStr

class SecretBox(Object):
    contents: State[str]
    is_open: State[bool]

    async def observe(self, *, channel=None, world, observer_id=None):
        if self.is_open:
            description = f"An open box containing: {self.contents}"
        else:
            description = "A closed box. You cannot see what is inside."
        duration = self._observe_duration(description, world, observer_id)
        return TimedStr(duration=duration, content=description)
```

Use `self._observe_duration(content, world, observer_id)` to compute observation time based on token count, object `read_speed`, and the observer's `read_speed_gain`.

## Overriding `transit()`

`transit()` is called each tick to advance the object's activity queue. Override it to add time-driven state evolution (decay, cooldowns, movement, etc.):

```python
class Candle(Object):
    burn_remaining: State[float]  # seconds of burn time left
    is_lit: State[bool]

    async def transit(self, world):
        if self.is_lit and self.burn_remaining > 0:
            elapsed = world.time - self._busy_until
            self.burn_remaining = max(0, self.burn_remaining - elapsed)
            if self.burn_remaining == 0:
                self.is_lit = False
        # Always call super to process the activity queue
        await super().transit(world)
```

## Nesting Objects

Objects can hold child objects via `register_object()`. Children appear in `state_dict()` and are accessible via attribute access:

```python
class Room(Object):
    temperature: State[float]

    def __init__(self, object_id: str):
        super().__init__(object_id=object_id)
        self.temperature = 22.0
        # Register a child object
        light = Light("room_light")
        self.register_object("light", light)
        # or simply self.light = Light("room_light")

# Access: room.light.is_on
# state_dict(): {"temperature": 22.0, "light": {"is_on": false, ...}}
```

Circular references are detected and rejected at registration time.

## Connecting to the World

Register objects in a `World` so Agents can interact with them:

```python
from universe.core.universe import World

world = World(name="house", description="A simple house simulation.")
world.register_object("front_door", Door("front_door"))
world.register_object("living_room", Room("living_room"))
```

Agents observe and act on objects through **Channels** configured in their attention system. See the Agent quickstart for details.

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `object_id` | `str` | required | Unique identifier for the object |
| `actions` | `list[Action]` | `[]` | Actions available on this object |
| `read_speed` | `float` | `10` | Observation speed in tokens/second |

## Class Hierarchy

```
Serializable          # state dict, child object registry
  -> Object           # actions, activities, observe/transit/act
       -> Agent       # LLM-driven observe-think-act loop
```
