"""End-to-End integration test for the Universe simulation framework.

This test exercises the entire stack from Config -> LLMClient -> Object -> Agent -> World
using a real LLM API call to maximize coverage with minimal test code.
"""

import warnings as warnings_mod
from unittest.mock import AsyncMock, patch

import pytest

from universe.core.config import Config
from universe.core.llm_client import LLMClient, LLMResult, ToolCall, BudgetWarning
from universe.core.object_ import (
    Object,
    Action,
    Params,
    State,
    Channel,
    TimedStatus,
    ActionExecutionStatus,
)
from universe.core.agent import Agent, Soul, Role, Mindset, SwitchMindsetToAction
from universe.core.agent.agent import SwitchMindsetToParams
from universe.core.universe import World


class ToggleParams(Params):
    """Parameters for toggle action - empty since toggle needs no params"""
    pass


class ToggleAction(Action["Light", ToggleParams]):
    """Action to toggle the light on/off"""

    name = "toggle_light"
    description = "Toggle the light on or off."

    async def execute(self, obj: "Light", params: ToggleParams, actor: Agent, world: World) -> TimedStatus:
        obj.is_on = not obj.is_on
        return TimedStatus(duration=0.5, status=ActionExecutionStatus.SUCCESS)


class InspectAction(Action["Light", Params]):
    """Action unique to one light — used to test single-channel target inference."""

    name = "inspect_light"
    description = "Inspect the light status."

    async def execute(self, obj: "Light", params: Params, actor: Agent, world: World) -> TimedStatus:
        return TimedStatus(duration=0.1, status=ActionExecutionStatus.SUCCESS)


class Light(Object):
    """A simple light object that can be toggled on/off"""

    is_on: State[bool]

    def __init__(self, object_id: str, *, extra_actions: list[Action] | None = None):
        super().__init__(
            object_id=object_id,
            actions=[ToggleAction()] + (extra_actions or []),
            read_speed=20,  # tokens/second when observed
        )
        self.is_on = False


@pytest.mark.asyncio
async def test_world_step_with_real_llm():
    """End-to-end test: Create Light, Agent with attention hierarchy, run world.step()

    This test exercises:
    - Config (singleton, loads config.yml)
    - LLMClient (singleton, real API call, caching, logging, validator)
    - GenericsMeta (Action[Light, Params] type extraction)
    - SingletonMeta (Config, LLMClient)
    - State / PrivateState / is_state_annotation
    - Serializable (setattr, getattr, state registration)
    - Object (init, observe, act, enqueue_activity, transit, tool_call_as_action)
    - Action (get_llm_tool_definition, execute, GetParams)
    - Channel / Activity / ActionExecutionContext
    - Agent (react — full observe/think/act pipeline)
    - Attention / Soul / Role / Mindset
    - World (step, time advancement, agent/object registration)
    - TimedStr
    """
    # Reset singletons for clean test state
    if hasattr(Config, "_instance"):
        delattr(Config, "_instance")
    if hasattr(LLMClient, "_instance"):
        delattr(LLMClient, "_instance")

    # Create world
    world = World(name="test_room", description="A test room with a light.")

    # Create and register light object
    light = Light("light_001")
    world.register_object("light_001", light)

    # Create and register agent
    agent = Agent("test_agent")
    world.register_object("test_agent", agent)

    # Build attention hierarchy: Soul -> Role -> Mindset
    channel = Channel(
        cognitive_target="room_light",
        target_id="light_001",
        allowed_actions=["toggle_light"],
        budget=2000,
    )

    mindset = Mindset(
        name="light_controller",
        description="Control the room light based on needs.",
        model=None,  # Will use default from config
    )
    mindset.add_channel(channel)

    role = Role(
        name="assistant",
        description="A helpful assistant that manages room lighting.",
    )
    role.add_mindset(mindset)

    soul = Soul(
        name="helpful",
        description="A helpful and proactive personality.",
    )
    soul.add_role(role)

    agent.add_soul(soul)

    # Set active attention path
    agent.attention.current_soul = "helpful"
    agent.attention.current_role = "assistant"
    agent.attention.current_mindset = "light_controller"

    # Verify initial state
    initial_time = world.time
    assert initial_time == 0.0
    assert light.is_on is False
    assert len(light.activities) == 0

    # Record agent state before step
    agent_busy_before = agent.busy_until

    # Run one simulation step with real LLM
    # This exercises the full pipeline: observe -> think (LLM) -> act
    await world.step()

    # Assertions that verify the full pipeline executed:
    # 1. Time advanced (world.step() increments time)
    assert world.time > initial_time, "World time should have advanced"

    # 2. Pipeline verification: either the light was affected OR the agent's state changed
    # The LLM may choose not to toggle the light, but the pipeline should have run
    light_affected = light.is_on is True or len(light.activities) > 0 or light.busy_until > 0
    agent_affected = agent.busy_until > agent_busy_before or agent.busy_until > 0
    pipeline_executed = light_affected or agent_affected
    assert pipeline_executed, (
        f"LLM pipeline should have executed. "
        f"light_affected={light_affected} (is_on={light.is_on}, activities={len(light.activities)}, busy={light.busy_until}), "
        f"agent_affected={agent_affected} (busy_before={agent_busy_before}, busy_after={agent.busy_until})"
    )

    # 3. Verify the agent's attention system was properly configured
    assert agent.attention.get_current_soul().name == "helpful"
    assert agent.attention.get_current_role().name == "assistant"
    assert agent.attention.get_current_mindset().name == "light_controller"
    assert "room_light" in agent.attention.get_current_channels()

    # 4. Verify Config singleton was accessed
    config = Config()
    assert config is not None
    assert len(config.enabled_llms) > 0

    # 5. Verify LLMClient singleton was accessed
    llm_client = LLMClient()
    assert llm_client is not None


@pytest.mark.asyncio
async def test_react_with_mocked_llm_response():
    """Test Agent.react() with mocked LLM to cover _parse_response,
    multi-channel target injection, budget warning, and the Act phase.

    Covers agent.py lines: 77-80, 112-113, 123-138, 147-172, 216-226.
    """
    # light_a has an extra InspectAction (unique to it)
    light_a = Light("light_a", extra_actions=[InspectAction()])
    light_b = Light("light_b")

    world = World(name="mock_test", description="Mocked LLM test")
    world.register_object("light_a", light_a)
    world.register_object("light_b", light_b)

    agent = Agent("mock_agent")
    world.register_object("mock_agent", agent)

    # Two channels with same toggle_light action → triggers _inject_target_param
    # channel_a also has inspect_light (unique) and a tiny budget → BudgetWarning
    channel_a = Channel(
        cognitive_target="light_a",
        target_id="light_a",
        allowed_actions=["toggle_light", "inspect_light"],
        budget=1,  # tiny budget triggers BudgetWarning
    )
    channel_b = Channel(
        cognitive_target="light_b",
        target_id="light_b",
        allowed_actions=["toggle_light"],
        budget=2000,
    )

    mindset = Mindset(name="default", description="Test")
    mindset.add_channel(channel_a)
    mindset.add_channel(channel_b)

    role = Role(name="tester", description="Test")
    role.add_mindset(mindset)

    soul = Soul(name="s", description="Test")
    soul.add_role(role)

    agent.add_soul(soul)
    agent.attention.current_soul = "s"
    agent.attention.current_role = "tester"
    agent.attention.current_mindset = "default"

    # Mock LLM returns two tool calls:
    #  1. toggle_light with explicit target (multi-channel path)
    #  2. inspect_light without target (single-channel inference path)
    mock_result = LLMResult(
        thought="Toggle A and inspect it",
        tool_calls=[
            ToolCall(name="toggle_light", arguments={"target": "light_a"}),
            ToolCall(name="inspect_light", arguments={}),
        ],
        duration=0.5,
    )

    with warnings_mod.catch_warnings(record=True) as w:
        warnings_mod.simplefilter("always")
        with patch("universe.core.agent.agent._llm_client") as mock_llm:
            mock_llm.complete = AsyncMock(return_value=mock_result)
            await agent.react(world)

    # Budget warning fired for channel_a (budget=1 token, observation exceeds it)
    assert any(issubclass(x.category, BudgetWarning) for x in w)

    # Both tool calls targeted light_a → it should have activities enqueued
    assert len(light_a.activities) > 0
    # light_b was not targeted
    assert len(light_b.activities) == 0


@pytest.mark.asyncio
async def test_switch_mindset_action():
    """Test SwitchMindsetToAction execution and dynamic param schema.

    Covers agent.py lines: 261-266, 274-280.
    """
    world = World(name="switch_test", description="Test")
    agent = Agent("switcher")
    world.register_object("switcher", agent)

    m1 = Mindset(name="calm", description="Calm mode")
    m2 = Mindset(name="alert", description="Alert mode")
    role = Role(name="guard", description="Guard role")
    role.add_mindset(m1)
    role.add_mindset(m2)
    soul = Soul(name="v", description="Vigilant")
    soul.add_role(role)
    agent.add_soul(soul)

    agent.attention.current_soul = "v"
    agent.attention.current_role = "guard"
    agent.attention.current_mindset = "calm"

    action = SwitchMindsetToAction()

    # Successful switch
    result = await action.execute(
        agent, SwitchMindsetToParams(mindset_name="alert"), agent, world,
    )
    assert result.status == ActionExecutionStatus.SUCCESS
    assert result.terminal is True
    assert agent.attention.current_mindset == "alert"

    # Invalid mindset → FAIL
    result = await action.execute(
        agent, SwitchMindsetToParams(mindset_name="nonexistent"), agent, world,
    )
    assert result.status == ActionExecutionStatus.FAIL

    # Dynamic param schema lists available mindsets
    ch = Channel(cognitive_target="self", target_id="switcher", budget=2000)
    schema = SwitchMindsetToParams.param_json_schema(ch, world)
    assert set(schema["properties"]["mindset_name"]["enum"]) == {"calm", "alert"}


def test_agent_attention_helpers():
    """Cover Agent's add/get/remove delegation methods.

    Covers agent.py lines: 232, 235, 238, 241, 244, 247, 250, 253.
    """
    agent = Agent("helper_test")

    soul = Soul(name="s1", description="S1")
    role = Role(name="r1", description="R1")
    mindset = Mindset(name="m1", description="M1")

    agent.add_soul(soul)
    agent.add_role("s1", role)
    agent.add_mindset("s1", "r1", mindset)

    assert agent.get_soul("s1").name == "s1"
    assert agent.get_role("s1", "r1").name == "r1"
    assert agent.get_mindset("s1", "r1", "m1").name == "m1"

    assert agent.remove_mindset("s1", "r1", "m1").name == "m1"
    assert agent.remove_role("s1", "r1").name == "r1"
    assert agent.remove_soul("s1").name == "s1"
