# Bug Report: Flask Config.from_prefixed_env Crashes When Simple and Nested Keys Share Prefix

**Target**: `flask.config.Config.from_prefixed_env`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `from_prefixed_env` method crashes with a `TypeError` when environment variables create a conflict between a simple key and a nested key structure due to alphabetical processing order enforcing simple keys before nested ones.

## Property-Based Test

```python
import os
import json
from hypothesis import given, strategies as st
from flask.config import Config

@given(
    prefix=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=65, max_codepoint=90)),
    key=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=65, max_codepoint=90)),
    simple_value=st.integers(),
    nested_value=st.text()
)
def test_from_prefixed_env_conflicting_keys(prefix, key, simple_value, nested_value):
    config = Config(root_path='/')

    old_env = os.environ.copy()
    try:
        # Clear any existing prefixed env vars
        for k in list(os.environ.keys()):
            if k.startswith(f'{prefix}_'):
                del os.environ[k]

        os.environ[f'{prefix}_{key}'] = json.dumps(simple_value)
        os.environ[f'{prefix}_{key}__NESTED'] = json.dumps(nested_value)

        config.from_prefixed_env(prefix=prefix)

        assert isinstance(config.get(key), dict) or key not in config
    finally:
        os.environ.clear()
        os.environ.update(old_env)

# Run the test
test_from_prefixed_env_conflicting_keys()
```

<details>

<summary>
**Failing input**: `prefix='A', key='A', simple_value=0, nested_value=''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 33, in <module>
    test_from_prefixed_env_conflicting_keys()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 7, in test_from_prefixed_env_conflicting_keys
    prefix=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=65, max_codepoint=90)),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 25, in test_from_prefixed_env_conflicting_keys
    config.from_prefixed_env(prefix=prefix)
    ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/flask/config.py", line 183, in from_prefixed_env
    current[tail] = value
    ~~~~~~~^^^^^^
TypeError: 'int' object does not support item assignment
Falsifying example: test_from_prefixed_env_conflicting_keys(
    # The test always failed when commented parts were varied together.
    prefix='A',  # or any other generated value
    key='A',  # or any other generated value
    simple_value=0,  # or any other generated value
    nested_value='',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import os
import json
from flask.config import Config

# Clear environment to ensure clean test
for key in list(os.environ.keys()):
    if key.startswith('FLASK_'):
        del os.environ[key]

config = Config(root_path='/')

os.environ['FLASK_DB'] = json.dumps("sqlite")
os.environ['FLASK_DB__NAME'] = json.dumps("mydb")

config.from_prefixed_env(prefix="FLASK")
print("Config:", dict(config))
```

<details>

<summary>
TypeError: 'str' object does not support item assignment
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/repo.py", line 15, in <module>
    config.from_prefixed_env(prefix="FLASK")
    ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/flask/config.py", line 183, in from_prefixed_env
    current[tail] = value
    ~~~~~~~^^^^^^
TypeError: 'str' object does not support item assignment
```
</details>

## Why This Is A Bug

The `from_prefixed_env` method's docstring explicitly states that "Keys are loaded in :func:`sorted` order" and "Specific items in nested dicts can be set by separating the keys with double underscores (`__`)". It also promises that "If an intermediate key doesn't exist, it will be initialized to an empty dict."

However, the implementation violates these expectations in a specific but common scenario. When environment variables are processed in sorted order (as documented), a simple key like `FLASK_DB` is processed before its nested counterpart `FLASK_DB__NAME`. The code correctly sets `config['DB'] = "sqlite"` for the simple key. But when processing the nested key, the code at lines 176-183 assumes that any existing intermediate key must be a dictionary that can be traversed:

```python
for part in parts:
    if part not in current:  # Only checks existence, not type
        current[part] = {}
    current = current[part]  # Assumes current[part] is dict-like
current[tail] = value  # Crashes here when current is a string/int
```

This crashes because `current['DB']` is a string ("sqlite"), not a dictionary. The code only checks if the key exists (`if part not in current`), but doesn't verify that the existing value is a dictionary that can be traversed for nested keys.

## Relevant Context

This bug affects real-world usage patterns. It's common to have configuration that evolves from simple values to nested structures. For example:
- Starting with `FLASK_DATABASE="sqlite://db.sqlite"`
- Later adding `FLASK_DATABASE__POOL_SIZE=10` for connection pooling
- The alphabetical ordering means the simple key always wins and crashes the nested key processing

The Flask documentation at https://flask.palletsprojects.com/en/stable/api/#flask.Config.from_prefixed_env shows examples of nested keys but doesn't warn about this limitation. The sorted order processing is explicitly mentioned, suggesting the developers were aware of ordering dependencies but didn't handle this case.

## Proposed Fix

```diff
--- a/flask/config.py
+++ b/flask/config.py
@@ -174,8 +174,11 @@ class Config(dict):
         *parts, tail = key.split("__")

         for part in parts:
-            # If an intermediate dict does not exist, create it.
-            if part not in current:
+            # If an intermediate dict does not exist, or if the value
+            # at this key is not a dict, create/replace it with an empty dict.
+            if part not in current or not isinstance(current[part], dict):
+                # Overwrite non-dict values to allow nested key traversal
+                # This prioritizes nested structure over simple values
                 current[part] = {}

             current = current[part]
```