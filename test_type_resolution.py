"""Verify how get_type_hints resolves State[T] in Python 3.12."""
from typing import get_type_hints, get_origin, get_args, Annotated

from universe.core.object_ import Object, State, PrivateState
from universe.core.object_.state import _STATE_MARKER, is_state_annotation


class TestObject(Object):
    visible_field: State[str]


# Get the hint directly from get_type_hints
hints = get_type_hints(TestObject, include_extras=True)
hint = hints['visible_field']

print(f"hint = {hint}")
print(f"type(hint) = {type(hint)}")
print(f"get_origin(hint) = {get_origin(hint)}")
print(f"get_origin(hint) is State = {get_origin(hint) is State}")
print(f"get_origin(hint) is Annotated = {get_origin(hint) is Annotated}")
print(f"get_args(hint) = {get_args(hint)}")
print()
print(f"is_state_annotation(hint) = {is_state_annotation(hint)}")

# Let's also check what happens if we manually check the Annotated path
print()
print("Checking if Annotated path would work:")
if get_origin(hint) is Annotated:
    args = get_args(hint)
    print(f"  args = {args}")
    print(f"  args[1:] = {args[1:]}")
    print(f"  any(isinstance(a, type(_STATE_MARKER)) for a in args[1:]) = {any(isinstance(a, type(_STATE_MARKER)) for a in args[1:])}")
else:
    print(f"  get_origin(hint) is not Annotated, so Annotated path wouldn't trigger")
