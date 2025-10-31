# Bug Report: llm.default_plugins.openai_models not_nulls Function

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `not_nulls` function has a misleading type signature that claims to accept a `dict` but only works with Pydantic BaseModel instances that yield `(key, value)` tuples when iterated. When called with an actual Python dict, it raises a `ValueError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import not_nulls

@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_with_dict(data):
    """Property: not_nulls should filter out None values from a dict"""
    result = not_nulls(data)
    assert all(v is not None for v in result.values())
```

**Failing input**: `{"a": 1, "b": None}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import not_nulls

data = {"temperature": 0.7, "max_tokens": None, "seed": 42}
result = not_nulls(data)
```

**Output**:
```
ValueError: not enough values to unpack (expected 2, got 1)
```

## Why This Is A Bug

The function signature `def not_nulls(data) -> dict:` suggests it accepts a dictionary, and the function is documented as a general utility. However, the implementation `for key, value in data` only works with objects that yield tuples when iterated (like Pydantic v2 BaseModel instances), not with standard Python dicts.

When iterating over a Python dict without `.items()`, you only get keys, not `(key, value)` tuples. Trying to unpack a single key into two variables causes a ValueError.

The function works in its current usage (line 658: `kwargs = dict(not_nulls(prompt.options))`) because `prompt.options` is a Pydantic model, but the misleading signature and implementation violate the principle of least surprise and make the code fragile.

## Fix

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -913,7 +913,7 @@ def redact_data(input_dict):


 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    return {key: value for key, value in data.items() if value is not None}
```

Alternatively, if the function is intended only for Pydantic models, update the type hint:

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -1,6 +1,7 @@
 from llm import AsyncKeyModel, EmbeddingModel, KeyModel, hookimpl
 import llm
 from llm.utils import (
+from typing import Union
+from pydantic import BaseModel

-def not_nulls(data) -> dict:
+def not_nulls(data: Union[dict, BaseModel]) -> dict:
+    if isinstance(data, dict):
+        return {key: value for key, value in data.items() if value is not None}
     return {key: value for key, value in data if value is not None}
```