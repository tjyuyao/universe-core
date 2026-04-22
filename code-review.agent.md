# Code Review: Universe Project

A simulation framework where LLM-powered Agents interact with Objects in a World through an observe-think-act loop with logical time. Interesting design. Here are my findings:

---

## DESIGN: translate() is a naive string-replace

`src/universe/core/translate/__init__.py` — Uses sequential `str.replace()` which is order-dependent and can produce unintended results if one translation's output contains another translation's input. For error messages this is fine, but worth noting if the dictionary grows.

---

## STYLE / MINOR

- No tests — pytest-asyncio and pytest-cov are in dev dependencies but no test files exist

---

## DESIGN: observation noise

The default `observe()` method exposes the full agent `state_dict()`: `read_speed_gain`, `think_speed_gain`, `activities`, `_busy_until`, plus all child objects. The attention structure is buried in this.

Options: override `observe()` on the Agent subclass to surface only attention-relevant state, or set `channel.budget` to cap token usage. A cleaner method is to extend the State[T] type hint pattern to allow for private state fields. 
