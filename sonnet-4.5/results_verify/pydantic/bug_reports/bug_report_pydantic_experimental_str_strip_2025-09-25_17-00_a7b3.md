# Bug Report: pydantic.experimental.pipeline str_strip Inconsistent With Python's str.strip()

**Target**: `pydantic.experimental.pipeline.str_strip()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `str_strip()` method in pydantic's experimental pipeline API doesn't strip the same whitespace characters as Python's `str.strip()`. Specifically, it fails to strip ASCII separator characters `\x1c`, `\x1d`, `\x1e`, and `\x1f` that Python considers whitespace.

## Property-Based Test

```python
from typing import Annotated
from hypothesis import given, strategies as st
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as

@given(st.text())
def test_string_transformation_equivalence(text):
    """Test that pipeline string methods behave like the corresponding str methods."""

    class TestModel(BaseModel):
        strip_field: Annotated[str, validate_as(str).str_strip()]

    model = TestModel(strip_field=text)

    assert model.strip_field == text.strip()
```

**Failing input**: `'\x1f'` (and also `'\x1c'`, `'\x1d'`, `'\x1e'`)

## Reproducing the Bug

```python
from typing import Annotated
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as

text = '\x1f'

class TestModel(BaseModel):
    field: Annotated[str, validate_as(str).str_strip()]

model = TestModel(field=text)

print(f"Python's str.strip(): {repr(text.strip())}")
print(f"Pipeline's str_strip(): {repr(model.field)}")

assert model.field == text.strip()
```

Output:
```
Python's str.strip(): ''
Pipeline's str_strip(): '\x1f'
AssertionError
```

## Why This Is A Bug

The `str_strip()` method is defined as `return self.transform(str.strip)`, which sets the expectation that it will behave identically to Python's built-in `str.strip()` method.

However, the implementation in `_apply_transform()` (lines 428-431 of pipeline.py) optimizes this call by using pydantic_core's `strip_whitespace=True` schema option instead of the actual `str.strip()` function:

```python
if s['type'] == 'str':
    if func is str.strip:
        s = s.copy()
        s['strip_whitespace'] = True
        return s
```

The problem is that pydantic_core's whitespace definition differs from Python's. Python's `str.strip()` removes characters where `char.isspace()` returns True, which includes:
- `\x1c` (File Separator)
- `\x1d` (Group Separator)
- `\x1e` (Record Separator)
- `\x1f` (Unit Separator)

But pydantic_core's `strip_whitespace` only handles the common whitespace characters (space, tab, newline, etc.) and does not strip these separator characters.

## Fix

Remove the optimization for `str.strip` in `_apply_transform()` to use the actual Python `str.strip()` function, or ensure pydantic_core's `strip_whitespace` matches Python's whitespace semantics.

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -425,11 +425,6 @@ def _apply_transform(
     if s is None:
         return cs.no_info_plain_validator_function(func)

-    if s['type'] == 'str':
-        if func is str.strip:
-            s = s.copy()
-            s['strip_whitespace'] = True
-            return s
-        elif func is str.lower:
+    if s['type'] == 'str':
+        if func is str.lower:
             s = s.copy()
             s['to_lower'] = True
             return s
```

Alternatively, if keeping the optimization is important, pydantic_core should be updated to match Python's whitespace definition.