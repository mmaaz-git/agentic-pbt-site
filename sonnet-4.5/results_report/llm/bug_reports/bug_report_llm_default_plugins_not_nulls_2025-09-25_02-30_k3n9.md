# Bug Report: llm.default_plugins.openai_models not_nulls Function

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_nulls()` function crashes with a `ValueError: too many values to unpack` when called with a dictionary because it attempts to iterate over the dictionary directly instead of calling `.items()`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import not_nulls


@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_removes_none_values(data):
    result = not_nulls(data)
    for key, value in result.items():
        assert value is not None
```

**Failing input**: `{'a': 1}` (even the simplest dictionary fails)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import not_nulls

test_dict = {'a': 1, 'b': None, 'c': 'test'}
result = not_nulls(test_dict)
```

Running this code produces:
```
ValueError: too many values to unpack (expected 2)
```

## Why This Is A Bug

The function is called on line 658 in `build_kwargs()` with `prompt.options`, which is expected to be a dictionary or dict-like object. The current implementation tries to unpack dictionary keys as (key, value) tuples:

```python
def not_nulls(data) -> dict:
    return {key: value for key, value in data if value is not None}
```

When you iterate over a dictionary directly with `for key, value in data`, Python attempts to unpack each key (which is a single value) into two variables, causing a `ValueError`. The function should iterate over `data.items()` to get (key, value) tuples.

This bug affects every call to any OpenAI model's `execute()` method when options are provided, making the entire OpenAI plugin unusable with options.

## Fix

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -913,7 +913,7 @@ class Completion(Chat):


 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    return {key: value for key, value in data.items() if value is not None}
```