# Bug Report: llm.default_plugins.openai_models not_nulls Function Incompatible with Pydantic v2

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_nulls` function in `openai_models.py` is incompatible with Pydantic v2's BaseModel iteration behavior, causing a `ValueError` when attempting to filter None values from model options.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from pydantic import BaseModel
from typing import Optional
from hypothesis import given, strategies as st

class MinimalOptions(BaseModel):
    value1: Optional[int] = None
    value2: Optional[str] = None

def not_nulls(data):
    return {key: value for key, value in data if value is not None}

@given(
    st.builds(
        MinimalOptions,
        value1=st.one_of(st.none(), st.integers()),
        value2=st.one_of(st.none(), st.text())
    )
)
def test_not_nulls_fails_with_pydantic_models(options):
    try:
        result = not_nulls(options)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not enough values to unpack" in str(e)
```

**Failing input**: Any Pydantic BaseModel instance (e.g., `MinimalOptions(value1=42)`)

## Reproducing the Bug

```python
from pydantic import BaseModel
from typing import Optional

class Options(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 100

def not_nulls(data):
    return {key: value for key, value in data if value is not None}

options = Options(max_tokens=50)

result = not_nulls(options)
```

**Error**: `ValueError: not enough values to unpack (expected 2, got 1)`

## Why This Is A Bug

In Pydantic v2, iterating over a `BaseModel` instance yields field names (strings), not `(key, value)` tuples. The `not_nulls` function attempts to unpack each iterated item into `key, value`, which fails because each item is a single string.

The function is called at line 658 as:
```python
kwargs = dict(not_nulls(prompt.options))
```

Where `prompt.options` is a Pydantic BaseModel instance. This call will fail whenever the code path is executed.

**Impact**: This bug affects all OpenAI model executions that have non-default options, causing them to crash. The severity is high because it prevents core functionality from working.

## Fix

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -913,7 +913,7 @@ class Completion(Chat):


 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    return {key: value for key, value in data.model_dump().items() if value is not None}
```

Alternatively, since the function already returns a dict, the call site at line 658 could be simplified to just `kwargs = not_nulls(prompt.options)` instead of `kwargs = dict(not_nulls(prompt.options))`.