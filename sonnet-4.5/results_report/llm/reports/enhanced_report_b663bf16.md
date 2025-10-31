# Bug Report: llm.default_plugins.openai_models.not_nulls Type Mismatch with Dict Input

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `not_nulls` function fails with a ValueError when called with a regular Python dictionary, despite having no type hints restricting its input. The function only works with Pydantic BaseModel instances that yield `(key, value)` tuples during iteration.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for not_nulls function"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import not_nulls

@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_with_dict(data):
    """Property: not_nulls should filter out None values from a dict"""
    result = not_nulls(data)
    assert all(v is not None for v in result.values())

if __name__ == "__main__":
    # Run the test
    test_not_nulls_with_dict()
```

<details>

<summary>
**Failing input**: `{'': None}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 18, in <module>
    test_not_nulls_with_dict()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 11, in test_not_nulls_with_dict
    def test_not_nulls_with_dict(data):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 13, in test_not_nulls_with_dict
    result = not_nulls(data)
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py", line 916, in not_nulls
    return {key: value for key, value in data if value is not None}
                           ^^^^^^^^^^
ValueError: not enough values to unpack (expected 2, got 0)
Falsifying example: test_not_nulls_with_dict(
    data={'': None},
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction case for not_nulls bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import not_nulls

# Test case with regular Python dict
data = {"temperature": 0.7, "max_tokens": None, "seed": 42}
print(f"Input dict: {data}")
print("Calling not_nulls(data)...")

try:
    result = not_nulls(data)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError when calling not_nulls with dict
</summary>
```
Input dict: {'temperature': 0.7, 'max_tokens': None, 'seed': 42}
Calling not_nulls(data)...
Error: ValueError: too many values to unpack (expected 2)
```
</details>

## Why This Is A Bug

The `not_nulls` function at line 915-916 of `/llm/default_plugins/openai_models.py` uses the pattern `for key, value in data` which assumes that iterating over `data` yields `(key, value)` tuples. This works with Pydantic v2 BaseModel instances (which is how it's currently used in production at line 658), but fails with regular Python dictionaries.

When iterating over a standard Python dict without calling `.items()`, you only get the keys as single values, not `(key, value)` tuples. Attempting to unpack a single string key into two variables (`key, value`) causes a ValueError. For example, with the key `"temperature"` (11 characters), Python tries to unpack 11 characters into 2 variables, resulting in "too many values to unpack (expected 2)". With an empty string key `""`, it results in "not enough values to unpack (expected 2, got 0)".

The function has no docstring, no type hints on its parameter, and a generic name that suggests it should work as a general utility for filtering None values from dictionary-like objects. The return type hint `-> dict` further reinforces this expectation. This violates the principle of least surprise and makes the code fragile to future changes.

## Relevant Context

The function is currently only used once in the codebase at line 658:
```python
kwargs = dict(not_nulls(prompt.options))
```

Where `prompt.options` is an instance of `llm.Options`, which inherits from Pydantic's BaseModel. When Pydantic v2 BaseModel instances are iterated, they yield `(field_name, field_value)` tuples, which is why the current implementation works in production.

Testing confirms this behavior difference:
- Iterating a Pydantic model: yields `('temperature', 0.7)`, `('max_tokens', None)`, etc.
- Iterating a regular dict: yields `'temperature'`, `'max_tokens'`, etc. (just the keys)

The bug doesn't affect current users but creates technical debt and could cause issues if:
1. Someone tries to reuse this utility function elsewhere
2. The codebase is refactored to use regular dicts instead of Pydantic models
3. The function is exposed as part of a public API

## Proposed Fix

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -913,7 +913,10 @@ def redact_data(input_dict):


 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    # Handle both Pydantic models (which yield tuples) and regular dicts
+    if hasattr(data, 'items'):
+        return {key: value for key, value in data.items() if value is not None}
+    return {key: value for key, value in data if value is not None}
```

Alternative minimal fix:

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -913,7 +913,7 @@ def redact_data(input_dict):


 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    return {key: value for key, value in dict(data).items() if value is not None}
```