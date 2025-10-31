# Bug Report: llm.default_plugins.openai_models not_nulls Dict Iteration

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `not_nulls` function incorrectly iterates over its input, using dict-unpacking syntax that only works with Pydantic models, not regular Python dicts. This contradicts the function's type signature and makes it unusable as a general utility.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import not_nulls

@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_filters_none_values(d):
    result = not_nulls(d)
    assert all(v is not None for k, v in result.items())
```

**Failing input**: `{'': None}` (or any dict)

## Reproducing the Bug

```python
from llm.default_plugins.openai_models import not_nulls

data = {'temperature': 0.5, 'max_tokens': None}
result = not_nulls(data)
```

**Output**:
```
ValueError: too many values to unpack (expected 2)
```

## Why This Is A Bug

The function signature `def not_nulls(data) -> dict:` implies it accepts any dict-like object. The implementation uses `for key, value in data` which:
- Works with Pydantic BaseModel instances (which iterate as tuples)
- Fails with regular Python dicts (which iterate as keys only)

While the current usage at line 658 (`kwargs = dict(not_nulls(prompt.options))`) works because `prompt.options` is a Pydantic model, this is still a bug because:
1. The type hint is misleading
2. The function cannot be reused for general dict filtering
3. It violates the principle of least surprise

## Fix

```diff
def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    return {key: value for key, value in data.items() if value is not None}
```

This fix makes the function work with both regular dicts and any object that has an `.items()` method (including Pydantic models via `model_dump().items()` if needed).