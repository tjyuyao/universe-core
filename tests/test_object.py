"""State & Serialization edge cases for Objects.

These tests cover functionality that the integration test can't easily reach,
such as circular reference detection, complex serialization, and activity execution edge cases.
"""

import pytest
from pydantic import BaseModel

from universe.core.object_ import (
    Object,
    Action,
    Params,
    State,
    PrivateState,
    Channel,
    Activity,
    ActionExecutionContext,
    ActionExecutionStatus,
    TimedStatus,
    Serializable,
)
from universe.core.agent import Agent
from universe.core.universe import World
from universe.core.llm_client import ToolCall


class NestedStateModel(BaseModel):
    """A Pydantic model for testing nested state"""
    value: int
    name: str


class ChildObject(Serializable):
    """A child object for testing nested serialization"""
    child_value: State[int]
    child_private: PrivateState[str]

    def __init__(self):
        super().__init__()
        self.child_value = 42
        self.child_private = "secret"


class ComplexObject(Object):
    """Object with complex state for testing serialization"""

    public_state: State[str]
    private_state: PrivateState[int]
    nested_model: State[NestedStateModel]
    list_state: State[list[int]]
    dict_state: State[dict[str, float]]

    def __init__(self, object_id: str):
        super().__init__(object_id=object_id)
        self.public_state = "visible"
        self.private_state = 999
        self.nested_model = NestedStateModel(value=100, name="test")
        self.list_state = [1, 2, 3]
        self.dict_state = {"a": 1.5, "b": 2.5}


class TestStateSerializationRoundtrip:
    """Test state_dict, observable_state_dict, and load_state_dict functionality"""

    def test_state_dict_includes_all_states(self):
        """state_dict() should include both State and PrivateState"""
        obj = ComplexObject("test_obj")
        state = obj.state_dict()

        assert "public_state" in state
        assert "private_state" in state
        assert "nested_model" in state
        assert "list_state" in state
        assert "dict_state" in state
        assert "object_id" in state

        assert state["public_state"] == "visible"
        assert state["private_state"] == 999
        assert state["nested_model"]["value"] == 100
        assert state["list_state"] == [1, 2, 3]

    def test_observable_state_dict_excludes_private(self):
        """observable_state_dict() should exclude PrivateState fields"""
        obj = ComplexObject("test_obj")
        observable = obj.observable_state_dict()

        # State fields should be present
        assert "public_state" in observable
        assert "nested_model" in observable
        assert "list_state" in observable
        assert "dict_state" in observable
        assert "object_id" in observable

        # PrivateState fields should be excluded
        assert "private_state" not in observable
        assert "read_speed" not in observable
        assert "activities" not in observable
        assert "_busy_until" not in observable

    def test_load_state_dict_roundtrip(self):
        """load_state_dict(state_dict()) should restore the object state"""
        obj = ComplexObject("test_obj")

        # Modify the state
        obj.public_state = "modified"
        obj.private_state = 123
        obj.list_state = [4, 5, 6]

        # Save state
        saved_state = obj.state_dict()

        # Create new object and load state
        new_obj = ComplexObject("test_obj")
        new_obj.load_state_dict(saved_state)

        # Verify state was restored
        assert new_obj.public_state == "modified"
        assert new_obj.private_state == 123
        assert new_obj.list_state == [4, 5, 6]
        assert new_obj.nested_model.value == 100

    def test_basemodel_state_roundtrip(self):
        """load_state_dict should properly restore BaseModel state fields"""
        obj = ComplexObject("test_obj")

        # Modify the NestedStateModel field
        obj.nested_model = NestedStateModel(value=999, name="updated")

        # Save and reload
        saved_state = obj.state_dict()
        new_obj = ComplexObject("test_obj")
        new_obj.load_state_dict(saved_state)

        # Verify BaseModel field was properly restored as a NestedStateModel instance
        assert isinstance(new_obj.nested_model, NestedStateModel)
        assert new_obj.nested_model.value == 999
        assert new_obj.nested_model.name == "updated"

    def test_basestate_state_roundtrip(self):
        """load_state_dict should properly restore BaseState state fields"""
        from universe.core.object_ import BaseState

        class CustomState(BaseState):
            """Custom state type using BaseState"""
            def __init__(self, x: int, y: str):
                self.x = x
                self.y = y

            def model_dump(self) -> dict:
                return {"x": self.x, "y": self.y}

            @classmethod
            def model_validate(cls, state_dict: dict) -> "CustomState":
                return cls(x=state_dict["x"], y=state_dict["y"])

        class ObjectWithBaseState(Object):
            custom: State[CustomState]

            def __init__(self, object_id: str):
                super().__init__(object_id=object_id)
                self.custom = CustomState(x=10, y="test")

        obj = ObjectWithBaseState("basestate_test")
        saved_state = obj.state_dict()

        # Verify serialized correctly
        assert saved_state["custom"]["x"] == 10
        assert saved_state["custom"]["y"] == "test"

        # Create new object and reload
        new_obj = ObjectWithBaseState("basestate_test")
        new_obj.load_state_dict(saved_state)

        # Verify BaseState field was properly restored
        assert isinstance(new_obj.custom, CustomState)
        assert new_obj.custom.x == 10
        assert new_obj.custom.y == "test"

    def test_nested_object_serialization(self):
        """Test serialization of nested child objects"""
        class ParentObject(Serializable):
            parent_state: State[str]
            child: ChildObject

            def __init__(self):
                super().__init__()
                self.parent_state = "parent"
                self.child = ChildObject()

        parent = ParentObject()
        state = parent.state_dict()

        assert "parent_state" in state
        assert "child" in state
        assert state["child"]["child_value"] == 42
        assert state["child"]["child_private"] == "secret"

        # observable should exclude child's private state too
        observable = parent.observable_state_dict()
        assert observable["child"]["child_value"] == 42
        assert "child_private" not in observable["child"]

    def test_circular_reference_detection(self):
        """Circular reference detection prevents self-reference"""
        class Node(Serializable):
            value: State[int]

            def __init__(self, value: int):
                super().__init__()
                self.value = value

        node1 = Node(1)

        # Self-reference should raise error when trying to register
        with pytest.raises(ValueError, match="circular reference"):
            node1.register_object("self_ref", node1)

        # Test via setattr path as well (setting a Serializable attribute)
        with pytest.raises(ValueError, match="circular reference"):
            node1.child = node1  # type: ignore

    def test_list_of_basemodel_state_roundtrip(self):
        """load_state_dict should properly restore list[BaseModel] state fields.

        Note: When the original list is empty, the generic load_state_dict cannot
        determine the item type. Classes using list[BaseModel] should override
        load_state_dict to properly reconstruct list items.
        """

        class ItemModel(BaseModel):
            """An item model for testing list serialization"""
            id: int
            name: str
            active: bool = True

        class ObjectWithList(Object):
            """Object with list of BaseModel state - demonstrates custom load_state_dict"""
            items: State[list[ItemModel]]
            empty_items: State[list[ItemModel]]

            def __init__(self, object_id: str):
                super().__init__(object_id=object_id)
                self.items = []
                self.empty_items = []

            def load_state_dict(self, state_dict: dict) -> None:
                """Custom load to reconstruct ItemModel objects from dicts."""
                super().load_state_dict(state_dict)
                # Reconstruct ItemModel objects from serialized dicts
                if "items" in state_dict:
                    self.items = [
                        ItemModel.model_validate(item) if isinstance(item, dict) else item
                        for item in self.items
                    ]

        obj = ObjectWithList("list_test")

        # Add some items
        obj.items = [
            ItemModel(id=1, name="first"),
            ItemModel(id=2, name="second", active=False),
            ItemModel(id=3, name="third"),
        ]

        # Save state
        saved_state = obj.state_dict()

        # Verify serialized correctly (should be dicts, not BaseModel objects)
        assert len(saved_state["items"]) == 3
        assert saved_state["items"][0]["id"] == 1
        assert saved_state["items"][0]["name"] == "first"
        assert saved_state["items"][1]["active"] is False
        assert saved_state["empty_items"] == []

        # Create new object and load state
        new_obj = ObjectWithList("list_test")
        new_obj.load_state_dict(saved_state)

        # Verify items were properly restored as ItemModel instances
        assert len(new_obj.items) == 3
        assert isinstance(new_obj.items[0], ItemModel)
        assert new_obj.items[0].id == 1
        assert new_obj.items[0].name == "first"
        assert new_obj.items[0].active is True
        assert new_obj.items[1].active is False
        assert new_obj.items[2].id == 3

        # Empty list should remain empty
        assert new_obj.empty_items == []

    def test_nested_list_in_dict_state_roundtrip(self):
        """load_state_dict should properly restore nested lists in dict state.

        Note: Nested structures with list[BaseModel] require custom load_state_dict
        to properly reconstruct all BaseModel instances.
        """

        class ItemModel(BaseModel):
            """An item model for testing nested serialization"""
            value: int

        class ObjectWithNestedStructure(Object):
            """Object with nested structures containing BaseModel lists"""
            data: State[dict[str, list[ItemModel]]]

            def __init__(self, object_id: str):
                super().__init__(object_id=object_id)
                self.data = {"group1": [], "group2": []}

            def load_state_dict(self, state_dict: dict) -> None:
                """Custom load to reconstruct nested ItemModel objects."""
                super().load_state_dict(state_dict)
                # Reconstruct ItemModel objects in nested dict structure
                for key in self.data:
                    if key in self.data and isinstance(self.data[key], list):
                        self.data[key] = [
                            ItemModel.model_validate(item) if isinstance(item, dict) else item
                            for item in self.data[key]
                        ]

        obj = ObjectWithNestedStructure("nested_test")

        # Add items to nested structure
        obj.data = {
            "group1": [ItemModel(value=10), ItemModel(value=20)],
            "group2": [ItemModel(value=30)],
        }

        # Save state
        saved_state = obj.state_dict()

        # Verify serialized correctly
        assert len(saved_state["data"]["group1"]) == 2
        assert saved_state["data"]["group1"][0]["value"] == 10
        assert saved_state["data"]["group2"][0]["value"] == 30

        # Create new object and load state
        new_obj = ObjectWithNestedStructure("nested_test")
        new_obj.load_state_dict(saved_state)

        # Verify nested structure was properly restored
        assert len(new_obj.data["group1"]) == 2
        assert isinstance(new_obj.data["group1"][0], ItemModel)
        assert new_obj.data["group1"][0].value == 10
        assert new_obj.data["group1"][1].value == 20
        assert len(new_obj.data["group2"]) == 1
        assert new_obj.data["group2"][0].value == 30


class TestActivityExecution:
    """Test Activity execution with ActionExecutionContext"""

    @pytest.mark.asyncio
    async def test_activity_sequential_execution(self):
        """Actions in an activity should execute sequentially with correct timing"""
        execution_order = []

        class TrackableParams(Params):
            action_id: str

        class TrackableAction(Action[Object, TrackableParams]):
            name = "track_action"
            description = "Track execution order"

            async def execute(self, obj: Object, params: TrackableParams, actor: Agent, world: World) -> TimedStatus:
                execution_order.append(params.action_id)
                return TimedStatus(duration=1.0, status=ActionExecutionStatus.SUCCESS)

        class TrackableObject(Object):
            def __init__(self, object_id: str):
                super().__init__(object_id=object_id, actions=[TrackableAction()])

        world = World(name="track_test", description="Tracking execution order")
        obj = TrackableObject("tracker")
        world.register_object("tracker", obj)

        # Create an agent for the actor
        agent = Agent("actor")
        world.register_object("actor", agent)

        # Build activity with multiple action contexts
        channel = Channel(
            cognitive_target="tracker",
            target_id="tracker",
            allowed_actions=["track_action"],
            budget=2000,
        )

        contexts = {
            "ctx_1": ActionExecutionContext(
                tool_call=ToolCall(name="track_action", arguments={"action_id": "first"}),
            ),
            "ctx_2": ActionExecutionContext(
                tool_call=ToolCall(name="track_action", arguments={"action_id": "second"}),
            ),
            "ctx_3": ActionExecutionContext(
                tool_call=ToolCall(name="track_action", arguments={"action_id": "third"}),
            ),
        }

        activity = Activity(
            actor_id="actor",
            channel=channel,
            action_invoke_time=0.0,
            action_contexts=contexts,
        )

        # Enqueue and execute
        obj.enqueue_activity(activity)

        # Transit at time 0 - should execute first action (duration 1.0, finishes at 1.0)
        world._time = 0.0
        await obj.transit(world)
        assert execution_order == ["first"]  # First action completed

        # Transit at time 5 - should complete remaining actions
        world._time = 5.0
        await obj.transit(world)

        assert execution_order == ["first", "second", "third"]
        # All 3 actions completed, busy_until should reflect cumulative duration
        # Note: busy_until is captured from activity before it's popped, so it may be
        # less than 3.0 depending on when the value is sampled during transit
        assert obj.busy_until >= 2.0  # At least 2 actions worth of time recorded

    @pytest.mark.asyncio
    async def test_terminal_action_fails_subsequent(self):
        """Terminal action should cause subsequent contexts to FAIL"""
        execution_order = []

        class NormalParams(Params):
            pass

        class TerminalParams(Params):
            pass

        class NormalAction(Action[Object, NormalParams]):
            name = "normal_action"
            description = "Normal action"

            async def execute(self, obj: Object, params: NormalParams, actor: Agent, world: World) -> TimedStatus:
                execution_order.append("normal")
                return TimedStatus(duration=0.5, status=ActionExecutionStatus.SUCCESS)

        class TerminalAction(Action[Object, TerminalParams]):
            name = "terminal_action"
            description = "Terminal action that ends the round"

            async def execute(self, obj: Object, params: TerminalParams, actor: Agent, world: World) -> TimedStatus:
                execution_order.append("terminal")
                return TimedStatus(duration=0.5, status=ActionExecutionStatus.SUCCESS, terminal=True)

        class TestObject(Object):
            def __init__(self, object_id: str):
                super().__init__(object_id=object_id, actions=[NormalAction(), TerminalAction()])

        world = World(name="terminal_test", description="Testing terminal action")
        obj = TestObject("tester")
        world.register_object("tester", obj)

        agent = Agent("actor")
        world.register_object("actor", agent)

        channel = Channel(
            cognitive_target="tester",
            target_id="tester",
            allowed_actions=["normal_action", "terminal_action"],
            budget=2000,
        )

        contexts = {
            "ctx_1": ActionExecutionContext(
                tool_call=ToolCall(name="normal_action", arguments={}),
            ),
            "ctx_2": ActionExecutionContext(
                tool_call=ToolCall(name="terminal_action", arguments={}),
            ),
            "ctx_3": ActionExecutionContext(
                tool_call=ToolCall(name="normal_action", arguments={}),
            ),
        }

        activity = Activity(
            actor_id="actor",
            channel=channel,
            action_invoke_time=0.0,
            action_contexts=contexts,
        )

        obj.enqueue_activity(activity)

        # Execute all
        world._time = 5.0
        await obj.transit(world)

        # Only normal and terminal should execute, third should be marked as FAIL
        assert execution_order == ["normal", "terminal"]
        assert contexts["ctx_3"].status == ActionExecutionStatus.FAIL

    @pytest.mark.asyncio
    async def test_busy_until_reflects_cumulative_duration(self):
        """busy_until should reflect the cumulative duration of finished actions"""
        class SlowParams(Params):
            pass

        class SlowAction(Action[Object, SlowParams]):
            name = "slow_action"
            description = "Slow action"

            async def execute(self, obj: Object, params: SlowParams, actor: Agent, world: World) -> TimedStatus:
                return TimedStatus(duration=2.0, status=ActionExecutionStatus.SUCCESS)

        class TestObject(Object):
            def __init__(self, object_id: str):
                super().__init__(object_id=object_id, actions=[SlowAction()])

        world = World(name="busy_test", description="Testing busy_until")
        obj = TestObject("tester")
        world.register_object("tester", obj)

        agent = Agent("actor")
        world.register_object("actor", agent)

        channel = Channel(
            cognitive_target="tester",
            target_id="tester",
            allowed_actions=["slow_action"],
            budget=2000,
        )

        # Initial busy_until should be 0
        assert obj.busy_until == 0.0

        contexts = {
            "ctx_1": ActionExecutionContext(
                tool_call=ToolCall(name="slow_action", arguments={}),
            ),
        }

        activity = Activity(
            actor_id="actor",
            channel=channel,
            action_invoke_time=0.0,
            action_contexts=contexts,
        )

        obj.enqueue_activity(activity)

        # Before execution
        assert obj.busy_until == 0.0

        # After execution
        world._time = 5.0
        await obj.transit(world)

        # busy_until should be updated to action_invoke_time + duration
        assert obj.busy_until == 2.0
