 ---
  Code Review: Universe Project

  A simulation framework where LLM-powered Agents interact with Objects in a World through an
  observe-think-act loop with logical time. Interesting design. Here are my findings:

  ---
  CRITICAL: Security - API Keys in Config

  config.yml — API keys are committed to git in plaintext (both active and commented-out). Even though some
  are commented out, they're in version history.

  - Move to environment variables or a .env file
  - Add config.yml to .gitignore and provide a config.yml.example instead
  - Rotate all exposed keys immediately

  ---
  BUG: Universe.__init__ mutable default argument

  src/universe/core/universe/universe.py:8:
  def __init__(self, name: str, worlds: dict[str, World] = {}):
  Classic Python mutable default argument trap. All instances share the same dict. Use None with a default
  inside the body.

  ---
  BUG: Action.GetParams incorrect call

  src/universe/core/object_/object_.py:35:
  return cls.GetParamsType().model_validate_json(**arguments)
  model_validate_json expects a single str argument (JSON string), not **kwargs. This should likely be
  model_validate(arguments) since arguments is already a dict at this point.

  ---
  BUG: LLMCache.get() signature mismatch with _generate_key()

  src/universe/core/llm_client/llm_cache.py:194 — get() accepts max_tokens parameter but doesn't pass it to
  _generate_key() and _generate_key() doesn't accept it either. The max_tokens param is dead code. The
  docstring mentions it but it has no effect.

  ---
  DESIGN: Serializable.__setattr__ catches too broadly

  src/universe/core/object_/object_.py and serializable.py — The State metaclass makes isinstance(x, State)
  return True for str, int, float, bool, list, dict, None, and BaseModel. This means almost every attribute
  assignment goes through register_state() rather than normal __setattr__. Properties like object_id,
  read_speed, capacity, actions (which is a dict) will all be treated as state variables. The actions dict
  specifically would be registered as a state and serialized, which likely isn't intended.

  ---
  DESIGN: translate() is a naive string-replace

  src/universe/core/translate/__init__.py — Uses sequential str.replace() which is order-dependent and can
  produce unintended results if one translation's output contains another translation's input. For error
  messages this is fine, but worth noting if the dictionary grows.

  ---
  STYLE / MINOR

  - No tests — pytest-asyncio and pytest-cov are in dev dependencies but no test files exist
  - mypy in runtime deps — pyproject.toml:13 lists mypy as a runtime dependency. It should only be in dev
  dependencies (it's already there too)
  - import hjson as json — Used in budget.py, serializable.py, object_.py. This silently changes JSON behavior
   (HJSON is a superset). If strict JSON is needed elsewhere, this aliasing could cause confusion. Consider
  importing as hjson explicitly
  - CacheEntry.created_at uses __import__ — llm_cache.py:35-36: field(default_factory=lambda:
  __import__('datetime').datetime.now().isoformat()) — Just import datetime at the top of the file normally

  ---
  Summary

  ┌──────────┬───────┬─────────────────────────────────────────────────────────────────────────────────────┐
  │ Severity │ Count │                                      Key Items                                      │
  ├──────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Critical │ 1     │ API keys in version control                                                         │
  ├──────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Bug      │ 5     │ UnboundLocalError in LLM complete, World.__init__ missing super, KeyError in step,  │
  │          │       │ mutable default, GetParams wrong call                                               │
  ├──────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Design   │ 3     │ Dead cleanup code, State metaclass too broad, shared singleton dict                 │
  ├──────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Minor    │ 4     │ No tests, mypy in runtime deps, hjson aliasing, __import__ trick                    │
  └──────────┴───────┴─────────────────────────────────────────────────────────────────────────────────────┘

  The core architecture (Object/Agent/World with logical time and channels) is well thought out. The bugs
  above should be straightforward to fix. The most impactful design issue is the State metaclass making
  isinstance too broad — it will likely cause surprising behavior as more attributes are added to Object
  subclasses.