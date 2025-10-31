# Bug Report: Flask Config.from_prefixed_env Type Collision

**Target**: `flask.Config.from_prefixed_env`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`Config.from_prefixed_env()` crashes with `TypeError` when environment variables define both a flat key and a nested key with the same prefix, because it attempts to assign dict values to non-dict types.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for Flask Config.from_prefixed_env type collision bug"""
from hypothesis import given, strategies as st, settings
from flask import Config
import tempfile
import os

# Strategy for valid environment variable keys (uppercase letters and underscores)
st_key = st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ_', min_size=1, max_size=20)

@settings(max_examples=100)
@given(
    key1=st_key.filter(lambda x: '__' not in x),
    value1=st.integers(),
    value2=st.integers()
)
def test_from_prefixed_env_collision_flat_then_nested(key1, value1, value2):
    """Test that from_prefixed_env handles flat+nested key collisions gracefully"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(tmpdir)

        env_key_flat = f"FLASK_{key1}"
        env_key_nested = f"FLASK_{key1}__SUBKEY"

        # Save old values if they exist
        old_flat = os.environ.get(env_key_flat)
        old_nested = os.environ.get(env_key_nested)

        try:
            # Set both flat and nested keys
            os.environ[env_key_flat] = str(value1)
            os.environ[env_key_nested] = str(value2)

            # This should either work or fail with a clear error
            config.from_prefixed_env()

            # If it works, verify the config is sensible
            assert key1 in config, f"Key {key1} should be in config"

        except TypeError as e:
            # This is the bug - TypeError is not a clear error
            print(f"\n❌ BUG FOUND with inputs: key1={key1!r}, value1={value1}, value2={value2}")
            print(f"   Environment variables: {env_key_flat}={value1}, {env_key_nested}={value2}")
            print(f"   Error: {e}")
            raise  # Re-raise to let Hypothesis catch it

        finally:
            # Restore original environment
            if old_flat is None:
                os.environ.pop(env_key_flat, None)
            else:
                os.environ[env_key_flat] = old_flat
            if old_nested is None:
                os.environ.pop(env_key_nested, None)
            else:
                os.environ[env_key_nested] = old_nested

if __name__ == "__main__":
    # Run the test
    print("Running property-based test for Flask Config.from_prefixed_env...")
    print("=" * 60)
    test_from_prefixed_env_collision_flat_then_nested()
```

<details>

<summary>
**Failing input**: `key1='A', value1=0, value2=0`
</summary>
```
Running property-based test for Flask Config.from_prefixed_env...
============================================================

❌ BUG FOUND with inputs: key1='A', value1=0, value2=0
   Environment variables: FLASK_A=0, FLASK_A__SUBKEY=0
   Error: 'int' object does not support item assignment

[Additional test runs showing the same error pattern across many different values...]

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 62, in <module>
    test_from_prefixed_env_collision_flat_then_nested()
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 12, in test_from_prefixed_env_collision_flat_then_nested
    @given(
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 35, in test_from_prefixed_env_collision_flat_then_nested
    config.from_prefixed_env()
  File "/home/npc/miniconda/lib/python3.13/site-packages/flask/config.py", line 183, in from_prefixed_env
    current[tail] = value
TypeError: 'int' object does not support item assignment
Falsifying example: test_from_prefixed_env_collision_flat_then_nested(
    key1='A',
    value1=0,
    value2=0,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of Flask Config.from_prefixed_env type collision bug"""
import os
import tempfile
from flask import Config

# Clean any existing FLASK_ env vars first
for key in list(os.environ.keys()):
    if key.startswith("FLASK_"):
        del os.environ[key]

with tempfile.TemporaryDirectory() as tmpdir:
    config = Config(tmpdir)

    # Setting both a flat key and a nested key with the same prefix
    os.environ["FLASK_DATABASE"] = "123"  # This becomes integer 123 via json.loads
    os.environ["FLASK_DATABASE__HOST"] = "localhost"  # This tries to set DATABASE['HOST']

    try:
        config.from_prefixed_env()
        print("Config loaded successfully:")
        print(f"  DATABASE = {config.get('DATABASE')}")
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")
        print(f"\nThis happens because:")
        print(f"  1. FLASK_DATABASE='123' is processed first (alphabetically)")
        print(f"  2. It gets parsed as integer 123 via json.loads()")
        print(f"  3. config['DATABASE'] = 123 is set")
        print(f"  4. FLASK_DATABASE__HOST is processed next")
        print(f"  5. Code tries to set config['DATABASE']['HOST'] = 'localhost'")
        print(f"  6. But config['DATABASE'] is 123 (int), not a dict!")
```

<details>

<summary>
Error occurred: TypeError: 'int' object does not support item assignment
</summary>
```
Error occurred: TypeError: 'int' object does not support item assignment

This happens because:
  1. FLASK_DATABASE='123' is processed first (alphabetically)
  2. It gets parsed as integer 123 via json.loads()
  3. config['DATABASE'] = 123 is set
  4. FLASK_DATABASE__HOST is processed next
  5. Code tries to set config['DATABASE']['HOST'] = 'localhost'
  6. But config['DATABASE'] is 123 (int), not a dict!
```
</details>

## Why This Is A Bug

This violates expected behavior because `from_prefixed_env()` should handle configuration conflicts gracefully rather than crashing with a cryptic error. The bug occurs due to the implementation in `/home/npc/miniconda/lib/python3.13/site-packages/flask/config.py` lines 154-183:

1. Environment variables are processed in sorted alphabetical order (line 154: `for key in sorted(os.environ)`)
2. When `FLASK_DATABASE` is encountered first, it's parsed with `json.loads()` and stored as integer 123 (lines 161-169)
3. When `FLASK_DATABASE__HOST` is processed, the code splits on `__` and attempts to traverse nested dicts (lines 173-183)
4. The code assumes intermediate keys are either non-existent or dict types, but doesn't check if they're already set to non-dict values
5. At line 181, `current = current[part]` retrieves the integer 123
6. At line 183, `current[tail] = value` attempts dict-like assignment on an integer, causing TypeError

The documentation states that "If an intermediate key doesn't exist, it will be initialized to an empty dict" but doesn't specify behavior when the intermediate key EXISTS with a non-dict value. This represents an undocumented edge case that results in a confusing crash rather than a clear error message explaining the configuration conflict.

## Relevant Context

The Flask documentation for `from_prefixed_env` (added in Flask 2.1) describes the double underscore mechanism for nested configuration but doesn't warn about potential conflicts between flat and nested keys. This is a realistic scenario that could occur when:

- Different team members set environment variables with conflicting structures
- Configuration is aggregated from multiple sources (e.g., Docker environment, CI/CD systems, cloud platform settings)
- Migration from flat to nested configuration structure is incomplete
- JSON values in environment variables collide with nested key patterns

The relevant code is in Flask's config.py: https://github.com/pallets/flask/blob/main/src/flask/config.py

## Proposed Fix

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
+                        f"{part!r} is already set to a non-dict value ({type(current[part]).__name__}). "
+                        f"Conflicting environment variables: FLASK_{part} and FLASK_{key}"
+                    )
                 # If an intermediate dict does not exist, create it.
                 if part not in current:
                     current[part] = {}
```