# Bug Report: Flask Config.from_prefixed_env Type Collision

**Target**: `flask.Config.from_prefixed_env`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`Config.from_prefixed_env()` crashes with `TypeError` when environment variables define both a flat key and a nested key with the same prefix (e.g., `FLASK_DATABASE=123` and `FLASK_DATABASE__HOST=localhost`).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from flask import Config
import tempfile
import os

st_key = st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ_', min_size=1, max_size=20)

@settings(max_examples=100)
@given(
    key1=st_key.filter(lambda x: '__' not in x),
    value1=st.integers(),
    value2=st.integers()
)
def test_from_prefixed_env_collision_flat_then_nested(key1, value1, value2):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(tmpdir)

        env_key_flat = f"FLASK_{key1}"
        env_key_nested = f"FLASK_{key1}__SUBKEY"

        old_flat = os.environ.get(env_key_flat)
        old_nested = os.environ.get(env_key_nested)
        try:
            os.environ[env_key_flat] = str(value1)
            os.environ[env_key_nested] = str(value2)
            config.from_prefixed_env()
        finally:
            if old_flat is None:
                os.environ.pop(env_key_flat, None)
            else:
                os.environ[env_key_flat] = old_flat
            if old_nested is None:
                os.environ.pop(env_key_nested, None)
            else:
                os.environ[env_key_nested] = old_nested
```

**Failing input**: `key1='A'`, `value1=0`, `value2=0`

## Reproducing the Bug

```python
import os
import tempfile
from flask import Config

with tempfile.TemporaryDirectory() as tmpdir:
    config = Config(tmpdir)

    os.environ["FLASK_DATABASE"] = "123"
    os.environ["FLASK_DATABASE__HOST"] = "localhost"

    config.from_prefixed_env()
```

**Error**:
```
TypeError: 'int' object does not support item assignment
```

## Why This Is A Bug

The bug occurs because environment variables are processed in sorted alphabetical order. When both `FLASK_DATABASE` and `FLASK_DATABASE__HOST` are set:

1. `FLASK_DATABASE` is processed first (alphabetically before `FLASK_DATABASE__HOST`)
2. The value is parsed as JSON: `"123"` → `123` (integer)
3. `config['DATABASE'] = 123` is set
4. Then `FLASK_DATABASE__HOST` is processed
5. The code tries to execute `config['DATABASE']['HOST'] = 'localhost'`
6. But `config['DATABASE']` is `123` (an integer), not a dict
7. Python raises `TypeError: 'int' object does not support item assignment`

This violates the expected behavior that `from_prefixed_env()` should either:
- Handle the collision gracefully (e.g., skip, warn, or override)
- Validate and reject conflicting configurations with a clear error message

The current behavior is an unhandled crash with a confusing error message that doesn't explain the real problem (conflicting environment variable definitions).

## Fix

```diff
--- a/flask/config.py
+++ b/flask/config.py
@@ -174,6 +174,13 @@ class Config(dict):  # type: ignore[type-arg]
             *parts, tail = key.split("__")

             for part in parts:
+                # Check if the key exists and is not a dict
+                if part in current and not isinstance(current[part], dict):
+                    raise ValueError(
+                        f"Cannot set nested config key {key!r}: "
+                        f"{part!r} is already set to a non-dict value. "
+                        f"Remove either FLASK_{part} or FLASK_{key}."
+                    )
                 # If an intermediate dict does not exist, create it.
                 if part not in current:
                     current[part] = {}
```

This fix detects the collision and raises a clear `ValueError` explaining the problem, rather than crashing with a confusing `TypeError`.