# Bug Report: llm.default_plugins.openai_models not_nulls Function

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_nulls()` function expects an iterable of (key, value) tuples but is called with `prompt.options` which can be a dict. When a non-empty dict is passed, Python's iteration yields keys only (not key-value pairs), causing a ValueError when the dict comprehension tries to unpack each key string into two variables.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from llm.default_plugins.openai_models import not_nulls

@settings(max_examples=100)
@given(st.dictionaries(
    st.text(min_size=1, max_size=20),
    st.one_of(st.none(), st.integers(), st.floats(allow_nan=False), st.text()),
    min_size=1
))
def test_not_nulls_fails_with_dict_argument(options_dict):
    try:
        result = not_nulls(options_dict)
        assert False, f"Expected ValueError but got result: {result}"
    except (ValueError, TypeError) as e:
        assert "unpack" in str(e) or "iterable" in str(e).lower()
```

**Failing input**: Any non-empty dict, e.g., `{"temperature": 0.7, "max_tokens": None}`

## Reproducing the Bug

```python
from llm.default_plugins.openai_models import not_nulls

options_dict = {"temperature": 0.7, "max_tokens": None, "top_p": 0.9}

try:
    result = not_nulls(options_dict)
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")
```

Expected output:
```
ValueError: not enough values to unpack (expected 2, got 1)
```

## Why This Is A Bug

The `not_nulls()` function is called at line 658 of `openai_models.py`:

```python
kwargs = dict(not_nulls(prompt.options))
```

However, `prompt.options` can be either:
1. An empty dict `{}` (when no options provided, see `models.py:365`)
2. A dict with option values
3. A Pydantic `Options` BaseModel instance

The function implementation at line 915:
```python
def not_nulls(data) -> dict:
    return {key: value for key, value in data if value is not None}
```

This dict comprehension expects `data` to be an iterable of (key, value) tuples. However:
- Iterating over a Python dict yields **keys only**, not (key, value) pairs
- For a non-empty dict, trying to unpack each key (a string) into `key, value` raises ValueError

The correct pattern is used elsewhere in the codebase (see `models.py:874`):
```python
for key, value in dict(self.prompt.options).items()
```

## Fix

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -655,7 +655,7 @@ class _Shared:
             return openai.OpenAI(**kwargs)

     def build_kwargs(self, prompt, stream):
-        kwargs = dict(not_nulls(prompt.options))
+        kwargs = not_nulls(dict(prompt.options).items())
         json_object = kwargs.pop("json_object", None)
         if "max_tokens" not in kwargs and self.default_max_tokens is not None:
             kwargs["max_tokens"] = self.default_max_tokens
```

Alternatively, fix the `not_nulls` function to handle dicts:

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -913,5 +913,8 @@ def combine_chunks(chunks: List) -> dict:


 def not_nulls(data) -> dict:
+    if isinstance(data, dict):
+        data = data.items()
     return {key: value for key, value in data if value is not None}
```

Or inline the logic (recommended):

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -655,7 +655,8 @@ class _Shared:
             return openai.OpenAI(**kwargs)

     def build_kwargs(self, prompt, stream):
-        kwargs = dict(not_nulls(prompt.options))
+        options_dict = dict(prompt.options) if not isinstance(prompt.options, dict) else prompt.options
+        kwargs = {k: v for k, v in options_dict.items() if v is not None}
         json_object = kwargs.pop("json_object", None)
         if "max_tokens" not in kwargs and self.default_max_tokens is not None:
             kwargs["max_tokens"] = self.default_max_tokens
@@ -912,6 +913,3 @@ def combine_chunks(chunks: List) -> dict:
     return combined

-
-def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
```