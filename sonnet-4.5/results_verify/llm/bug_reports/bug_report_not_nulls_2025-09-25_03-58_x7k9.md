# Bug Report: llm.default_plugins.openai_models.not_nulls Function

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_nulls` function crashes when called with its expected input types (dict or Pydantic model) due to incorrect iteration syntax. The function attempts to unpack key-value pairs from a dict/model by iterating directly over it (`for key, value in data`), but this syntax only works with iterables of tuples like `dict.items()`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import not_nulls

@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_with_dict(input_dict):
    result = dict(not_nulls(input_dict.items()))

    assert None not in result.values()
    for key, value in result.items():
        assert key in input_dict
        assert input_dict[key] == value
        assert value is not None
```

**Failing input**: Any dict, e.g., `{"a": 1, "b": None}`

## Reproducing the Bug

```python
from llm.default_plugins.openai_models import not_nulls

test_dict = {"a": 1, "b": None, "c": "hello"}

try:
    result = dict(not_nulls(test_dict))
except ValueError as e:
    print(f"Crashed with: {e}")
```

Expected behavior: Returns `{"a": 1, "c": "hello"}` (filters out None values)
Actual behavior: Raises `ValueError: not enough values to unpack (expected 2, got 1)`

The function is called at line 658:
```python
kwargs = dict(not_nulls(prompt.options))
```

Where `prompt.options` is a Pydantic `BaseModel` instance (lines 441, 479, 541 show it's created as `self.model.Options(**options)`).

## Why This Is A Bug

The function signature and usage indicate it should accept dict-like objects and filter out None values. However:

1. **With dicts**: Iterating `for key, value in some_dict` fails because iterating over a dict yields only keys, not (key, value) tuples. You need `.items()`.

2. **With Pydantic models**: Iterating over a Pydantic v2 `BaseModel` yields field names (strings), not (key, value) tuples. Attempting to unpack each field name into two variables fails.

3. **Usage context**: Line 658 calls `not_nulls(prompt.options)` where `prompt.options` is a Pydantic model instance, which would trigger this crash.

## Fix

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -912,5 +912,8 @@ class Completion(Chat):


 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    if hasattr(data, 'model_dump'):
+        data = data.model_dump()
+    if isinstance(data, dict):
+        data = data.items()
+    return {key: value for key, value in data if value is not None}
```

Alternatively, a simpler fix assuming the caller should pass `.items()`:

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -912,5 +912,5 @@ class Completion(Chat):


 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    return {key: value for key, value in data.items() if value is not None}
```

However, this requires updating the caller at line 658 to not call `.items()` (or the function needs to handle both cases).