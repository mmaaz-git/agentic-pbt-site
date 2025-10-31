# Bug Report: llm.default_plugins.openai_models.not_nulls ValueError

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_nulls()` function crashes when called with a Pydantic model (its intended input type) due to incorrect iteration syntax, causing all OpenAI model executions to fail.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic import BaseModel, Field
from typing import Optional

class SharedOptions(BaseModel):
    temperature: Optional[float] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    seed: Optional[int] = Field(default=42)

def not_nulls(data) -> dict:
    return {key: value for key, value in data if value is not None}

@given(
    st.one_of(st.none(), st.floats(min_value=0, max_value=2)),
    st.one_of(st.none(), st.integers(min_value=1)),
)
def test_not_nulls_filters_none_values(temperature, max_tokens):
    opts = SharedOptions(temperature=temperature, max_tokens=max_tokens)
    result = not_nulls(opts)

    for key, value in result.items():
        assert value is not None
```

**Failing input**: Any Pydantic model instance, e.g., `SharedOptions(temperature=0.5, max_tokens=100)`

## Reproducing the Bug

```python
from pydantic import BaseModel, Field
from typing import Optional

def not_nulls(data) -> dict:
    return {key: value for key, value in data if value is not None}

class SharedOptions(BaseModel):
    temperature: Optional[float] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)

opts = SharedOptions(temperature=0.5, max_tokens=100)
result = not_nulls(opts)
```

**Error**: `ValueError: too many values to unpack (expected 2, got N)` where N is the length of the field name string.

## Why This Is A Bug

The `not_nulls()` function is called in `build_kwargs()` at line 658 with `prompt.options`, which is a Pydantic `BaseModel` instance. When iterating over a Pydantic model with `for x in model`, it yields field names as strings (e.g., "temperature", "max_tokens"). The code `for key, value in data` attempts to unpack each string into two variables, which fails.

The function is in the critical path for all OpenAI model executions (both `Chat` and `AsyncChat` classes), making this a high-severity crash that would prevent any usage of the OpenAI models plugin.

## Fix

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -913,7 +913,7 @@ class Completion(Chat):


 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    return {key: value for key, value in data.items() if value is not None}
```