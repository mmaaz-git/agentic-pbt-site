# Bug Report: llm.default_plugins.openai_models.not_nulls - TypeError when called with Pydantic BaseModel

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_nulls` function crashes with a `TypeError` when called with a Pydantic v2 BaseModel instance because it attempts to unpack field names as `(key, value)` tuples, but Pydantic v2 iteration yields only field names.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import not_nulls, SharedOptions

@given(
    st.one_of(st.none(), st.floats(min_value=0, max_value=2, allow_nan=False)),
    st.one_of(st.none(), st.integers(min_value=1, max_value=10000))
)
def test_not_nulls_works_with_pydantic_model(temperature, max_tokens):
    options = SharedOptions(temperature=temperature, max_tokens=max_tokens)
    result = not_nulls(options)

    assert isinstance(result, dict)
    for key in result:
        assert result[key] is not None
```

**Failing input**: `SharedOptions(temperature=0.5, max_tokens=100)`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import not_nulls, SharedOptions

options = SharedOptions(temperature=0.5, max_tokens=100)
result = not_nulls(options)
```

**Expected**: Returns `{'temperature': 0.5, 'max_tokens': 100}`

**Actual**: Raises `TypeError: cannot unpack non-iterable str object` because iterating over a Pydantic BaseModel yields field names (strings), not `(key, value)` tuples.

## Why This Is A Bug

The function is called at line 658 of `openai_models.py`:

```python
kwargs = dict(not_nulls(prompt.options))
```

Where `prompt.options` is a Pydantic BaseModel instance created at various places in `models.py`:

```python
options=self.model.Options(**options)
```

The `not_nulls` function (line 915) attempts to iterate and unpack:

```python
def not_nulls(data) -> dict:
    return {key: value for key, value in data if value is not None}
```

In Pydantic v2 (which this library requires: `pydantic>=2.0.0`), iterating over a BaseModel yields field names only, not tuples. This causes a crash when trying to unpack a string into `(key, value)`.

## Fix

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -913,4 +913,7 @@ class Completion(Chat):

 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    if hasattr(data, 'model_dump'):
+        items = data.model_dump().items()
+    else:
+        items = data
+    return {key: value for key, value in items if value is not None}
```