# Agent Usage Quickstart

## Overview

`Agent` is a subclass of `Object` that adds an LLM-driven **observe-think-act** loop. Each tick, an Agent:

1. **Observe** -- reads the state of target Objects through Channels
2. **Think** -- sends the observations to an LLM, which decides what to do
3. **Act** -- parses tool calls from the LLM response and enqueues them as Activities on target Objects

Agents are configured through an **Attention** system that determines their personality (Soul), behavioral context (Role), and current focus (Mindset).

## Minimal Example

```python
from universe.core.object_ import Object, Action, Params, Channel, TimedStatus, ActionExecutionStatus, State
from universe.core.agent import Agent, Soul, Role, Mindset
from universe.core.universe import World


# 1. Define a target object with an action
class Light(Object):
    is_on: State[bool]

    def __init__(self, object_id: str):
        super().__init__(object_id=object_id, actions=[ToggleAction()])
        self.is_on = False

class ToggleAction(Action[Light, Params]):
    name = "toggle_light"
    description = "Toggle the light on or off."

    async def execute(self, obj, params, actor, world):
        obj.is_on = not obj.is_on
        return TimedStatus(duration=0.5, status=ActionExecutionStatus.SUCCESS)


# 2. Create the world and register objects
world = World(name="room", description="A room with a light.")
world.register_object("light", Light("light"))

# 3. Create an agent
agent = Agent("alice")
world.register_object("alice", agent)

# 4. Configure attention: Soul -> Role -> Mindset
channel = Channel(
    cognitive_target="the_light",   # alias the agent sees
    target_id="light",              # actual object ID in world
    allowed_actions=["toggle_light"],
)

mindset = Mindset(
    name="default",
    description="Focus on managing the room light.",
    model="gpt-4o-mini",            # LLM model name (from config.yml)
)
mindset.add_channel(channel)

role = Role(name="assistant", description="A helpful room assistant.")
role.add_mindset(mindset)

soul = Soul(name="friendly", description="A friendly personality.")
soul.add_role(role)

agent.add_soul(soul)

# 5. Set the active attention path
agent.attention.current_soul = "friendly"
agent.attention.current_role = "assistant"
agent.attention.current_mindset = "default"

# 6. Run the simulation
import asyncio
asyncio.run(world.step())
```

## Attention System

The Attention system is a three-level hierarchy that controls what the Agent perceives, how it behaves, and which LLM it uses:

```
Agent
  └── Attention
        ├── current_soul / current_role / current_mindset  (active path)
        └── souls: dict[str, Soul]
              └── Soul("friendly")
                    ├── description: personality traits
                    └── roles: dict[str, Role]
                          └── Role("assistant")
                                ├── description: behavioral context
                                └── mindsets: dict[str, Mindset]
                                      └── Mindset("default")
                                            ├── description: current focus
                                            ├── model: LLM model name
                                            └── channels: dict[str, Channel]
```

The active path (`current_soul` / `current_role` / `current_mindset`) determines which Mindset -- and therefore which Channels and LLM model -- the Agent uses during `react()`.

### Switching Context at Runtime

```python
# Switch to a different mindset
agent.attention.current_mindset = "combat"

# Switch to a different role entirely
agent.attention.current_role = "guard"
agent.attention.current_mindset = "patrol"
```

This enables dynamic behavior changes without creating new Agent instances.

### SwitchMindsetToAction

Agents have a built-in action `switch_mindset_to` that allows the LLM to switch mindsets autonomously:

```python
from universe.core.agent import SwitchMindsetToAction

Agent("alice", actions=[SwitchMindsetToAction()])

# self-channel:
Channel(cognitive_target="self", target_id="alice", allowed_actions=["switch_mindset_to"])

# The LLM can call: switch_mindset_to(mindset_name="combat")
# This switches to the "combat" mindset within the current role
```

Key characteristics:

- **Dynamic enum**: The `mindset_name` parameter shows only available mindsets in the current role
- **Must be last**: Should be the final tool call in a turn (subsequent calls are ignored)
- **5 second duration**: Switching mindset consumes 5 logical seconds
- **Fail on invalid**: Returns FAIL status if the mindset doesn't exist

## Channels

A **Channel** connects an Agent to a target Object. It defines:

| Field | Type | Description |
|-------|------|-------------|
| `cognitive_target` | `str` | Alias the agent perceives (e.g. `"the_door"`) |
| `target_id` | `str` | Actual object ID registered in the World |
| `budget` | `int \| None` | Max observation tokens (warns if exceeded) |
| `allowed_actions` | `list[str] \| None` | Whitelist of action names the agent can invoke |

```python
# Full observation, all actions allowed
open_channel = Channel(cognitive_target="lab", target_id="lab_room")

# Restricted: limited budget, only specific actions
restricted_channel = Channel(
    cognitive_target="locked_safe",
    target_id="safe",
    budget=500,
    allowed_actions=["inspect", "enter_code"],
)
```

When multiple Channels expose the same action name, the Agent's LLM tool definition is automatically augmented with a `target` parameter so the LLM can specify which object to act on.

## Agent Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_id` | `str` | required | Unique identifier (becomes `object_id`) |
| `actions` | `list[Action]` | `[]` | Actions others can perform on this agent |
| `read_speed` | `float` | `10` | How fast others read this agent's state (tokens/s) |
| `read_speed_gain` | `float` | `1.0` | Multiplier on how fast this agent reads others |
| `think_speed_gain` | `float` | `1.0` | Multiplier on LLM thinking speed |

## Logical Time

Every phase of the observe-think-act loop consumes logical time:

- **Observe**: `tokens / (object.read_speed * agent.read_speed_gain)` per Channel
- **Think**: `tokens / (base_think_speed * agent.think_speed_gain)` for LLM output
- **Act**: `duration` returned by each `Action.execute()`

An Agent cannot take new actions until its logical `busy_until` time is reached by the World clock. Faster agents (higher gain values) act more frequently in the simulation.

## Subclassing Agent

Since `Agent` extends `Object`, it can have its own state and actions -- allowing other Agents to observe and act on it:

```python
class ChatAgent(Agent):
    mood: State[str]
    inbox: State[list[str]]

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            actions=[SendMessageAction()],
        )
        self.mood = "neutral"
        self.inbox = []
```

## System Prompt Structure

The system prompt sent to the LLM is automatically constructed from the active attention path:

```
Your Identity: {agent_id}
World({world.name}): {world.description}
Soul({soul.name}): {soul.description}
Role({role.name}): {role.description}
Mindset({mindset.name}): {mindset.description}
```

The user prompt contains the serialized state of all observed Objects (via their `observe()` methods). You control what the LLM sees by overriding `observe()` on target Objects and by configuring Channel budgets.

## Simulation Loop

`World.step()` drives the simulation:

```python
# Run multiple ticks
for _ in range(100):
    await world.step()
```

Each step:
1. Advances the world clock by a small epsilon
2. Calls `agent.react(world)` for every Agent
3. Updates the world clock to `min(agent.busy_until for all agents)`

Agents that are still busy (logical time hasn't elapsed) skip their `react()` call and wait for the world clock to catch up.
