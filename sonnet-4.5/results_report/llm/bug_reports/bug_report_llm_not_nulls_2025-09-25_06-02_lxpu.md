# Bug Report: llm.default_plugins.openai_models.not_nulls Incorrect Dict Iteration

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_nulls` function crashes with a `ValueError` when called with any dictionary because it incorrectly iterates over the dictionary without calling `.items()`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import not_nulls

@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_removes_none_values(data):
    result = not_nulls(data)
    assert isinstance(result, dict)
    assert all(value is not None for value in result.values())
```

**Failing input**: `{'': None}`

## Reproducing the Bug

```python
from llm.default_plugins.openai_models import not_nulls

data = {'': None}
result = not_nulls(data)
```

This raises:
```
ValueError: not enough values to unpack (expected 2, got 0)
```

The error occurs because `for key, value in data` attempts to unpack dictionary keys (strings) into two variables, but dictionary iteration yields only keys, not (key, value) tuples.

## Why This Is A Bug

The function signature indicates it should accept any dict and filter out None values, but it crashes on all dictionary inputs. The function is used in production code (line 658 in `build_kwargs` method), where it's called as `not_nulls(prompt.options)`, making this a critical crash bug.

## Fix

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -913,4 +913,4 @@ class Completion(Chat):

 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    return {key: value for key, value in data.items() if value is not None}
```
