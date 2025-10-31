# Bug Report: pydantic.v1 Field `strip_whitespace` Parameter Has No Effect

**Target**: `pydantic.v1.Field(strip_whitespace=...)`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `strip_whitespace=True` parameter in `Field()` has no effect on string validation. While the parameter is accepted without error, it does not strip whitespace from string values, unlike the functionally equivalent `constr(strip_whitespace=True)` which works correctly.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.v1 import BaseModel, Field


class ModelWithStrip(BaseModel):
    value: str = Field(strip_whitespace=True)


@given(st.text())
def test_string_strip_whitespace(value):
    model = ModelWithStrip(value=value)
    assert model.value == value.strip()
```

**Failing input**: `value=' '` (and any string with leading/trailing whitespace)

## Reproducing the Bug

```python
from pydantic.v1 import BaseModel, Field, constr


class ModelWithStripField(BaseModel):
    value: str = Field(strip_whitespace=True)


class ModelWithStripConstr(BaseModel):
    value: constr(strip_whitespace=True)


value = ' hello '

model_field = ModelWithStripField(value=value)
print(f"Field(strip_whitespace=True): {repr(model_field.value)}")

model_constr = ModelWithStripConstr(value=value)
print(f"constr(strip_whitespace=True): {repr(model_constr.value)}")
```

Output:
```
Field(strip_whitespace=True): ' hello '
constr(strip_whitespace=True): 'hello'
```

The bug is that `Field(strip_whitespace=True)` does not strip the leading and trailing spaces from `' hello '`, while `constr(strip_whitespace=True)` correctly strips them to produce `'hello'`.

## Why This Is A Bug

This violates the API contract in two ways:

1. **Inconsistent behavior**: The same parameter name (`strip_whitespace`) behaves differently when used in `Field()` vs `constr()`. Users reasonably expect consistent behavior.

2. **Silent failure**: `Field(strip_whitespace=True)` accepts the parameter without any warning or error, but silently ignores it. This leads users to believe their validation is working when it is not.

The expected behavior is that `Field(strip_whitespace=True)` should strip leading and trailing whitespace from string values, matching the behavior of `constr(strip_whitespace=True)`.

## Fix

The fix should make `Field(strip_whitespace=True)` properly apply whitespace stripping. This likely involves ensuring that the `strip_whitespace` field info is properly processed during string validation:

```diff
--- a/pydantic/v1/validators.py
+++ b/pydantic/v1/validators.py
@@ -xxx,x +xxx,x @@ def str_validator(v: Any) -> str:
     ...existing validation...
+    # Apply strip_whitespace if specified in field info
+    if field and field.field_info and hasattr(field.field_info, 'strip_whitespace'):
+        if field.field_info.strip_whitespace:
+            v = v.strip()
     return v
```

Alternatively, the documentation should be updated to clearly state that `strip_whitespace` is only supported via `constr()` and not via `Field()`, and `Field()` should raise an error or warning when this parameter is used.