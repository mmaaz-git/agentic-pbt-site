# Bug Report: Flask Config.from_prefixed_env Crashes on Conflicting Keys

**Target**: `flask.config.Config.from_prefixed_env`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Config.from_prefixed_env` method crashes with a `TypeError` when environment variables create a conflict between a simple key and a nested key structure. For example, setting `FLASK_DB="value"` and `FLASK_DB__NAME="nested"` causes a crash because the code tries to traverse `DB` as a dictionary when it's already set to a simple string value.

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
        os.environ[f'{prefix}_{key}'] = json.dumps(simple_value)
        os.environ[f'{prefix}_{key}__NESTED'] = json.dumps(nested_value)

        config.from_prefixed_env(prefix=prefix)

        assert isinstance(config.get(key), dict) or key not in config
    finally:
        os.environ.clear()
        os.environ.update(old_env)
```

**Failing input**: `prefix="FLASK"`, `key="DB"`, `simple_value=42`, `nested_value="mydb"`

## Reproducing the Bug

```python
import os
import json
from flask.config import Config

config = Config(root_path='/')

os.environ['FLASK_DB'] = json.dumps("sqlite")
os.environ['FLASK_DB__NAME'] = json.dumps("mydb")

config.from_prefixed_env(prefix="FLASK")
```

**Output:**
```
Traceback (most recent call last):
  File "config.py", line 183, in from_prefixed_env
    current[tail] = value
    ^^^^^^^^^^^^^
TypeError: 'str' object does not support item assignment
```

## Why This Is A Bug

The docstring states that "Keys are loaded in :func:`sorted` order" and that "Specific items in nested dicts can be set by separating the keys with double underscores (`__`)". However, when a simple key like `FLASK_DB` is set before a nested key like `FLASK_DB__NAME` (which happens due to alphabetical sorting), the code crashes instead of handling the conflict gracefully.

The code at lines 176-183 assumes that if a key part exists in the config, it must be a dictionary when traversing nested paths:

```python
for part in parts:
    if part not in current:
        current[part] = {}

    current = current[part]  # Assumes current[part] is a dict

current[tail] = value  # Crashes if current is not a dict
```

This violates user expectations and makes the feature fragile when simple and nested keys share a prefix.

## Fix

```diff
--- a/flask/config.py
+++ b/flask/config.py
@@ -174,8 +174,13 @@ class Config(dict):  # type: ignore[type-arg]
         *parts, tail = key.split("__")

         for part in parts:
-            # If an intermediate dict does not exist, create it.
-            if part not in current:
+            # If an intermediate dict does not exist, or if the value
+            # at this key is not a dict, create/replace it with an empty dict.
+            if part not in current or not isinstance(current[part], dict):
+                # Overwrite non-dict values to allow nested key traversal
                 current[part] = {}

             current = current[part]
```