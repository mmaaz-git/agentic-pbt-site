# Bug Report: llm.default_plugins not_nulls Function Incorrect Iteration

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_nulls` function crashes when filtering model options because it attempts to iterate and unpack Pydantic model field names (or dict keys) as (key, value) tuples, causing a `ValueError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from llm.default_plugins.openai_models import not_nulls
from pydantic import BaseModel
from typing import Optional

class MockOptions(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

@given(st.floats(min_value=0, max_value=2), st.one_of(st.none(), st.integers(min_value=1)))
def test_not_nulls_with_pydantic_model(temp, tokens):
    options = MockOptions(temperature=temp, max_tokens=tokens)
    result = not_nulls(options)
    expected = {k: v for k, v in options.model_dump().items() if v is not None}
    assert result == expected
```

**Failing input**: `MockOptions(temperature=0.7, max_tokens=None)`

## Reproducing the Bug

```python
from pydantic import BaseModel
from typing import Optional

class Options(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

def not_nulls(data):
    return {key: value for key, value in data if value is not None}

options = Options(temperature=0.7, max_tokens=None)
result = not_nulls(options)
```

**Error**: `ValueError: too many values to unpack (expected 2)`

**Root Cause**:
- `prompt.options` is a Pydantic `BaseModel` instance (created on line 441: `options=self.model.Options(**options)`)
- When iterating `for key, value in data` over a Pydantic model, Python calls `iter(data)` which yields field names only (strings like `"temperature"`, `"max_tokens"`)
- Python then tries to unpack each field name string into two variables: `key, value = "temperature"`
- This fails because `"temperature"` has 11 characters, not 2

The same bug occurs if `prompt.options` is a dict (from line 365's fallback: `self.options = options or {}`), since iterating over a dict yields keys only.

## Why This Is A Bug

1. Called on **line 658** with `prompt.options` (a Pydantic model instance or dict)
2. **Line 916** iterates `for key, value in data` expecting (key, value) tuples
3. Crashes whenever options are provided with any field names longer than 2 characters
4. Affects all OpenAI model API calls that use options (temperature, max_tokens, etc.)
5. Function is only called once in entire codebase, making this a critical single point of failure

## Fix

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -913,4 +913,4 @@ def redact_data(input_dict):

 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    return {key: value for key, value in data.model_dump().items() if value is not None}
```

Or alternatively, fix the call site:

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -655,7 +655,7 @@ class _Shared:

     def build_kwargs(self, prompt, stream):
-        kwargs = dict(not_nulls(prompt.options))
+        kwargs = dict(not_nulls(prompt.options.model_dump().items()))
```