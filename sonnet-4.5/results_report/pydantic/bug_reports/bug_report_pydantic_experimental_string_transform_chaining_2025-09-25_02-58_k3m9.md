# Bug Report: pydantic.experimental.pipeline String Transform Chaining

**Target**: `pydantic.experimental.pipeline._apply_transform`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When chaining string transformations like `transform(str.lower).str_upper()`, only the first transformation is applied. The schema incorrectly sets both `to_lower: true` and `to_upper: true`, but pydantic-core only applies the first flag.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
from typing import Annotated

@given(st.text())
@settings(max_examples=200)
def test_str_lower_then_upper(text):
    class Model(BaseModel):
        value: Annotated[str, transform(str.lower).str_upper()]

    m = Model(value=text)
    assert m.value == text.upper()
```

**Failing input**: `text='A'`

## Reproducing the Bug

```python
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
from typing import Annotated

class Model(BaseModel):
    value: Annotated[str, transform(str.lower).str_upper()]

m = Model(value="ABC")
print(m.value)
```

Expected output: `ABC` (first apply `lower()` to get "abc", then apply `upper()` to get "ABC")
Actual output: `abc` (only `lower()` is applied)

## Why This Is A Bug

The documentation for `transform` states it should "Transform the output of the previous step." When chaining `transform(str.lower).str_upper()`, the expected behavior is:
1. Apply `str.lower` to get "abc"
2. Apply `str.upper` to the result to get "ABC"

However, the generated core schema has both `to_lower: true` and `to_upper: true` set simultaneously, and pydantic-core only applies `to_lower`.

## Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -461,15 +461,20 @@ def _apply_transform(
     if s is None:
         return cs.no_info_plain_validator_function(func)

-    if s['type'] == 'str':
+    # Only use schema-level optimizations if no conflicting transformation is already applied
+    if s['type'] == 'str' and not (s.get('to_lower') or s.get('to_upper')):
         if func is str.strip:
             s = s.copy()
             s['strip_whitespace'] = True
             return s
         elif func is str.lower:
             s = s.copy()
             s['to_lower'] = True
             return s
         elif func is str.upper:
             s = s.copy()
             s['to_upper'] = True
             return s

     return cs.no_info_after_validator_function(func, s)
```

This fix ensures that when a string transformation is already applied via schema flags (`to_lower` or `to_upper`), any subsequent string transformations use a validator function chain instead of setting conflicting flags.