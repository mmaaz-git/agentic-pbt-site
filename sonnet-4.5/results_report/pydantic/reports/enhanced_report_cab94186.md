# Bug Report: pydantic.v1 Field Multiple-Of Validation Fails for Large Integers

**Target**: `pydantic.v1.validators.number_multiple_validator`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `multiple_of` constraint in pydantic.v1 incorrectly validates large integers (>= ~10^16) due to floating-point precision loss, allowing non-multiples to pass validation.

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


if __name__ == "__main__":
    # Run the test with hypothesis
    try:
        test_multiple_of_constraint()
        print("Test passed for all examples")
    except AssertionError as e:
        print(f"Test failed: {e}")
```

<details>

<summary>
**Failing input**: `value=14_792_214_996_390_874`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 23, in <module>
    test_multiple_of_constraint()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 11, in test_multiple_of_constraint
    def test_multiple_of_constraint(value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 16, in test_multiple_of_constraint
    with pytest.raises(ValidationError):
         ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: DID NOT RAISE <class 'pydantic.v1.error_wrappers.ValidationError'>
Falsifying example: test_multiple_of_constraint(
    value=14_792_214_996_390_874,
)
```
</details>

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

<details>

<summary>
Output shows validation incorrectly passes for non-multiple
</summary>
```
Value: 100000000000000004
Is actually a multiple of 5: False
Remainder: 4
BUG: Model created with value 100000000000000004
Expected: ValidationError should have been raised
```
</details>

## Why This Is A Bug

The `multiple_of` constraint should enforce that integer values are exact multiples of the specified divisor. The value `100000000000000004` has remainder 4 when divided by 5 (`100000000000000004 % 5 == 4`), so it should fail the `multiple_of=5` constraint.

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/pydantic/v1/validators.py:182-183` where the validation logic uses:
```python
mod = float(v) / float(field_type.multiple_of) % 1
```

This approach converts integers to floats, causing precision loss for large values. IEEE 754 double-precision floats can only precisely represent integers up to 2^53 (approximately 9×10^15). Beyond this, not all integers can be exactly represented, leading to incorrect validation results.

For example:
- `14792214996390874 % 5 == 4` (correct integer arithmetic)
- But after float conversion, the validation incorrectly passes

## Relevant Context

- **Affected versions**: pydantic v1.10.19 (confirmed), likely affects all v1.x versions
- **Threshold**: Integers with absolute value >= ~10^16 are affected
- **Documentation**: The pydantic documentation states fields should be "a multiple of" the given number, with no mention of precision limitations
- **Use cases impacted**: Financial systems, scientific computing, cryptography, or any domain working with large integer identifiers
- **JSON Schema compliance**: This violates JSON Schema's `multipleOf` specification which requires exact divisibility

## Proposed Fix

Replace the floating-point division approach with integer modulo when both operands are integers:

```diff
--- a/pydantic/v1/validators.py
+++ b/pydantic/v1/validators.py
@@ -179,8 +179,14 @@ def number_multiple_validator(v: 'Number', field: 'ModelField') -> 'Number':
 def number_multiple_validator(v: 'Number', field: 'ModelField') -> 'Number':
     field_type: ConstrainedNumber = field.type_
     if field_type.multiple_of is not None:
-        mod = float(v) / float(field_type.multiple_of) % 1
-        if not almost_equal_floats(mod, 0.0) and not almost_equal_floats(mod, 1.0):
+        # Use integer arithmetic for exact validation when both are integers
+        if isinstance(v, int) and isinstance(field_type.multiple_of, int):
+            if v % field_type.multiple_of != 0:
+                raise errors.NumberNotMultipleError(multiple_of=field_type.multiple_of)
+        else:
+            # Fall back to float arithmetic for non-integers
+            mod = float(v) / float(field_type.multiple_of) % 1
+            if not almost_equal_floats(mod, 0.0) and not almost_equal_floats(mod, 1.0):
+                raise errors.NumberNotMultipleError(multiple_of=field_type.multiple_of)
     return v
```