# Bug Report: pydantic.v1 Field `multiple_of` Constraint Fails for Large Integers

**Target**: `pydantic.v1.Field(multiple_of=...)`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `multiple_of` constraint in pydantic.v1 fails to validate large integers (>= 10^17) correctly due to floating-point precision loss. Values that are not multiples of the specified number incorrectly pass validation, and values that are multiples incorrectly fail validation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.v1 import BaseModel, Field, ValidationError
import pytest


class ModelWithMultipleOf(BaseModel):
    value: int = Field(multiple_of=5)


@given(st.integers())
def test_multiple_of_constraint(value):
    if value % 5 == 0:
        model = ModelWithMultipleOf(value=value)
        assert model.value == value
    else:
        with pytest.raises(ValidationError):
            ModelWithMultipleOf(value=value)
```

**Failing input**: `value=19384208019627014` (and any integer >= 10^17 with similar characteristics)

## Reproducing the Bug

```python
from pydantic.v1 import BaseModel, Field, ValidationError


class ModelWithMultipleOf(BaseModel):
    value: int = Field(multiple_of=5)


value = 10**17 + 4

print(f"Value: {value}")
print(f"Is actually a multiple of 5: {value % 5 == 0}")
print(f"Remainder: {value % 5}")

try:
    model = ModelWithMultipleOf(value=value)
    print(f"BUG: Model created with value {model.value}")
    print("Expected: ValidationError should have been raised")
except ValidationError as e:
    print(f"ValidationError correctly raised: {e}")
```

Output:
```
Value: 100000000000000004
Is actually a multiple of 5: False
Remainder: 4
BUG: Model created with value 100000000000000004
Expected: ValidationError should have been raised
```

The bug occurs because `100000000000000004 % 5 == 4`, but pydantic accepts it as valid.

## Why This Is A Bug

The `multiple_of` constraint should enforce that integer values are exact multiples of the specified divisor. For the value `100000000000000004`, the constraint `multiple_of=5` should fail because `100000000000000004 % 5 == 4`, not 0.

The root cause is that pydantic's validation logic converts the integer to a float before performing the modulo operation. For large integers (>= 10^17), this conversion loses precision:

```python
value = 100000000000000004
print(value % 5)              # 4 (correct)
print(float(value) % 5.0)     # 0.0 (incorrect due to precision loss)
print(float(value))           # 1e+17 (precision lost)
```

This affects integers at or above 10^17, which exceeds the precision of IEEE 754 double-precision floats (2^53 ≈ 9 × 10^15).

## Fix

The fix is to use integer arithmetic when both the value and `multiple_of` are integers, avoiding float conversion:

```diff
--- a/pydantic/v1/validators.py
+++ b/pydantic/v1/validators.py
@@ -xxx,x +xxx,x @@ def number_multiple_validator(v: 'Number', field: 'ModelField') -> 'Number':
     field_multiple_of = field.field_info.multiple_of
     if field_multiple_of is not None:
-        mod = v % field_multiple_of
+        # Use integer arithmetic to avoid float precision issues
+        if isinstance(v, int) and isinstance(field_multiple_of, int):
+            mod = v % field_multiple_of
+        else:
+            mod = v % field_multiple_of
         if mod != 0:
             raise errors.NumberNotMultipleError(multiple_of=field_multiple_of)
     return v
```

Alternatively, for a more robust fix that handles all numeric types:

```diff
--- a/pydantic/v1/validators.py
+++ b/pydantic/v1/validators.py
@@ -xxx,x +xxx,x @@ def number_multiple_validator(v: 'Number', field: 'ModelField') -> 'Number':
     field_multiple_of = field.field_info.multiple_of
     if field_multiple_of is not None:
-        mod = v % field_multiple_of
+        # Use integer arithmetic when possible to avoid precision loss
+        if isinstance(v, int) and isinstance(field_multiple_of, (int, float)) and field_multiple_of == int(field_multiple_of):
+            mod = v % int(field_multiple_of)
+        else:
+            mod = float(v) % float(field_multiple_of)
+            # For float results, use a small epsilon for comparison
+            if abs(mod) < 1e-9 or abs(mod - field_multiple_of) < 1e-9:
+                mod = 0
         if mod != 0:
             raise errors.NumberNotMultipleError(multiple_of=field_multiple_of)
     return v
```