# Bug Report: llm.default_plugins.openai_models.not_nulls - Incorrect dict iteration

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_nulls` function attempts to iterate over a dict as if it were an iterable of (key, value) tuples, causing a `ValueError` when the function is called with a dict input. This function is used in `build_kwargs` and will crash whenever called.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_filters_none_values(d):
    result = not_nulls(d)

    assert isinstance(result, dict)
    for key, value in result.items():
        assert value is not None
```

**Failing input**: `{'a': 1}` (any non-empty dict causes the crash)

## Reproducing the Bug

```python
def not_nulls(data) -> dict:
    return {key: value for key, value in data if value is not None}

test_data = {'temperature': 0.7, 'max_tokens': None, 'top_p': 0.9}
result = not_nulls(test_data)
```

When run, this produces:
```
ValueError: not enough values to unpack (expected 2, got 1)
```

The bug occurs because:
1. `data` is a dict (confirmed by `prompt.options = options or {}` in llm/models.py:344)
2. Iterating over a dict with `for key, value in data` tries to unpack each key into two variables
3. Dict keys are single values (strings), not tuples, causing unpacking to fail

## Why This Is A Bug

This violates the function's contract and crashes on any dict input. The function is called from `build_kwargs` (line 658) as:
```python
kwargs = dict(not_nulls(prompt.options))
```

where `prompt.options` is explicitly a dict. This means `build_kwargs` will always crash, preventing any OpenAI model from being used.

## Fix

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -913,7 +913,7 @@ class Completion(Chat):


 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    return {key: value for key, value in data.items() if value is not None}


 def combine_chunks(chunks: List) -> dict:
```