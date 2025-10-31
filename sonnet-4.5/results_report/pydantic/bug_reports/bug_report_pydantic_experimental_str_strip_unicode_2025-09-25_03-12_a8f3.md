# Bug Report: pydantic.experimental.pipeline str_strip() Doesn't Strip Unicode Whitespace

**Target**: `pydantic.experimental.pipeline._Pipeline.str_strip`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `str_strip()` method in pipeline doesn't strip Unicode whitespace characters, only ASCII whitespace. This is inconsistent with Python's `str.strip()` behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
from typing import Annotated

@given(st.text())
@settings(max_examples=200)
def test_str_strip_matches_python(text):
    class Model(BaseModel):
        value: Annotated[str, validate_as(str).str_strip()]

    m = Model(value=text)
    assert m.value == text.strip()
```

**Failing input**: `text='\x1f'` (Unicode Unit Separator)

## Reproducing the Bug

```python
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
from typing import Annotated

class Model(BaseModel):
    value: Annotated[str, validate_as(str).str_strip()]

m = Model(value='\x1f')
print(f"Result: {m.value!r}")
print(f"Expected: {'\x1f'.strip()!r}")
```

Expected output: `''` (empty string, since `'\x1f'.strip()` returns `''`)
Actual output: `'\x1f'` (unchanged)

## Why This Is A Bug

The `str_strip()` method is defined as:

```python
def str_strip(self: _Pipeline[_InT, str]) -> _Pipeline[_InT, str]:
    return self.transform(str.strip)
```

This clearly indicates it should behave like Python's `str.strip()` method. However, the implementation optimizes it to use pydantic-core's `strip_whitespace` flag, which only strips ASCII whitespace (space, tab, newline, etc.) and not Unicode whitespace characters like `\x1f`.

Python's `str.strip()` removes all Unicode whitespace, including control characters classified as whitespace.

## Fix

The optimization in `_apply_transform` should be removed for `str.strip` to ensure correct behavior:

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -461,11 +461,6 @@ def _apply_transform(
     if s is None:
         return cs.no_info_plain_validator_function(func)

     if s['type'] == 'str':
-        if func is str.strip:
-            s = s.copy()
-            s['strip_whitespace'] = True
-            return s
-        elif func is str.lower:
+        if func is str.lower:
             s = s.copy()
             s['to_lower'] = True
             return s
```

Alternatively, the documentation should clarify that `str_strip()` only strips ASCII whitespace, and the method should be renamed to something like `str_strip_ascii()` to avoid confusion.