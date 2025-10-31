# Bug Report: pydantic.v1 Field multiple_of Constraint Fails for Large Integers

**Target**: `pydantic.v1.Field(multiple_of=...)`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `multiple_of` constraint in pydantic.v1 incorrectly accepts large integers that violate the constraint due to float precision loss in the validation logic.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.v1 import BaseModel, Field, ValidationError
import pytest

class MultipleOf(BaseModel):
    value: int = Field(multiple_of=5)

@given(st.integers().filter(lambda x: x % 5 != 0))
def test_multiple_of_rejects_invalid(value):
    with pytest.raises(ValidationError):
        MultipleOf(value=value)
```

**Failing input**: `17608513714555794` (and any integer >= 10^16 that is not a multiple of the specified value)

## Reproducing the Bug

```python
from pydantic.v1 import BaseModel, Field

class MultipleOf(BaseModel):
    value: int = Field(multiple_of=5)

value = 10**16 + 1

model = MultipleOf(value=value)
print(f"Accepted: {model.value}")
print(f"Value % 5 = {model.value % 5}")

assert model.value % 5 == 0
```

Output:
```
Accepted: 10000000000000001
Value % 5 = 1
AssertionError
```

## Why This Is A Bug

The validator in `pydantic.v1.validators.number_multiple_validator` converts integers to floats before checking the multiple_of constraint:

```python
mod = float(v) / float(field_type.multiple_of) % 1
```

For large integers (>= 10^16), this conversion loses precision because floats cannot exactly represent all integers beyond 2^53. This causes the validator to incorrectly compute the modulo, accepting values that violate the constraint.

## Fix

```diff
--- a/pydantic/v1/validators.py
+++ b/pydantic/v1/validators.py
@@ -182,7 +182,10 @@ def number_multiple_validator(v: 'Number', field: 'ModelField') -> 'Number':
 def number_multiple_validator(v: 'Number', field: 'ModelField') -> 'Number':
     field_type: ConstrainedNumber = field.type_
     if field_type.multiple_of is not None:
-        mod = float(v) / float(field_type.multiple_of) % 1
+        if isinstance(v, int) and isinstance(field_type.multiple_of, int):
+            mod = (v % field_type.multiple_of) / field_type.multiple_of
+        else:
+            mod = float(v) / float(field_type.multiple_of) % 1
         if not almost_equal_floats(mod, 0.0) and not almost_equal_floats(mod, 1.0):
             raise errors.NumberNotMultipleError(multiple_of=field_type.multiple_of)
     return v
```