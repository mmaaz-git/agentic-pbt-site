# Bug Report: pydantic.experimental.pipeline Gt Constraint Silently Fails with Type Mismatch

**Target**: `pydantic.experimental.pipeline._apply_constraint` (Gt constraint handling)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `gt()` (greater than) constraint in pydantic's experimental pipeline API completely fails to validate when the constraint value type doesn't match the field type (e.g., using `gt(5.5)` on an integer field), allowing all values to pass validation instead of properly rejecting invalid values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import validate_as
from typing import Annotated
import pytest


@given(st.integers(min_value=-1000, max_value=5))
@example(5)  # The boundary case that should fail
@example(0)  # A clear failure case
@example(-1000)  # Edge of the range
def test_gt_float_constraint_on_int(value):
    """Test that gt(5.5) rejects integers <= 5"""
    class Model(BaseModel):
        field: Annotated[int, validate_as(int).gt(5.5)]

    with pytest.raises(ValidationError):
        Model(field=value)


@given(st.integers(min_value=6, max_value=1000))
@example(6)  # The boundary case that should pass
@example(1000)  # Edge of the range
def test_gt_float_constraint_accepts_valid(value):
    """Test that gt(5.5) accepts integers > 5.5 (i.e., >= 6)"""
    class Model(BaseModel):
        field: Annotated[int, validate_as(int).gt(5.5)]

    m = Model(field=value)
    assert m.field == value
```

<details>

<summary>
**Failing input**: `value=5` (also fails for `value=0` and `value=-1000`)
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Running Hypothesis tests for gt(5.5) constraint on integer field...

  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 38, in <module>
  |     test_gt_float_constraint_on_int()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 9, in test_gt_float_constraint_on_int
  |     @example(5)  # The boundary case that should fail
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | BaseExceptionGroup: Hypothesis found 3 distinct failures in explicit examples. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 17, in test_gt_float_constraint_on_int
    |     with pytest.raises(ValidationError):
    |          ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    |     fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    |     ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    |     raise Failed(msg=reason, pytrace=pytrace)
    | Failed: DID NOT RAISE <class 'pydantic_core._pydantic_core.ValidationError'>
    | Falsifying explicit example: test_gt_float_constraint_on_int(
    |     value=5,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 17, in test_gt_float_constraint_on_int
    |     with pytest.raises(ValidationError):
    |          ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    |     fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    |     ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    |     raise Failed(msg=reason, pytrace=pytrace)
    | Failed: DID NOT RAISE <class 'pydantic_core._pydantic_core.ValidationError'>
    | Falsifying explicit example: test_gt_float_constraint_on_int(
    |     value=0,
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 17, in test_gt_float_constraint_on_int
    |     with pytest.raises(ValidationError):
    |          ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    |     fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    |     ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    |     raise Failed(msg=reason, pytrace=pytrace)
    | Failed: DID NOT RAISE <class 'pydantic_core._pydantic_core.ValidationError'>
    | Falsifying explicit example: test_gt_float_constraint_on_int(
    |     value=-1000,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import validate_as
from typing import Annotated


# Test gt(5.5) on integer field with value=5 (should fail but doesn't)
class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(5.5)]


print("Testing gt(5.5) with value=5 (should reject but doesn't):")
try:
    m = ModelGt(value=5)
    print(f"BUG: value=5 passed gt(5.5) validation! Result: {m.value}")
except ValidationError as e:
    print(f"Expected behavior: ValidationError raised")
    print(f"Error: {e}")

print("\n" + "="*60 + "\n")

# Test ge(5.5) on integer field with value=5 (should fail and does)
class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(5.5)]


print("Testing ge(5.5) with value=5 (should reject and does):")
try:
    m = ModelGe(value=5)
    print(f"BUG: value=5 passed ge(5.5) validation! Result: {m.value}")
except ValidationError as e:
    print(f"Correct: ge(5.5) properly rejects value=5")
    print(f"Error: {e}")

print("\n" + "="*60 + "\n")

# Test various values with gt(5.5)
print("Testing gt(5.5) with various integer values:")
test_values = [-1000, 0, 5, 6, 10]
for val in test_values:
    try:
        m = ModelGt(value=val)
        print(f"  value={val}: PASSED validation (result: {m.value})")
    except ValidationError:
        print(f"  value={val}: REJECTED by validation")

print("\n" + "="*60 + "\n")

# Demonstrating that 5 > 5.5 is False in Python
print("Mathematical check:")
print(f"  5 > 5.5 = {5 > 5.5}")
print(f"  6 > 5.5 = {6 > 5.5}")
```

<details>

<summary>
Output demonstrates gt(5.5) incorrectly accepting all values
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Testing gt(5.5) with value=5 (should reject but doesn't):
BUG: value=5 passed gt(5.5) validation! Result: 5

============================================================

Testing ge(5.5) with value=5 (should reject and does):
Correct: ge(5.5) properly rejects value=5
Error: 1 validation error for ModelGe
value
  Value error, Expected >= 5.5 [type=value_error, input_value=5, input_type=int]
    For further information visit https://errors.pydantic.dev/2.10/v/value_error

============================================================

Testing gt(5.5) with various integer values:
  value=-1000: PASSED validation (result: -1000)
  value=0: PASSED validation (result: 0)
  value=5: PASSED validation (result: 5)
  value=6: PASSED validation (result: 6)
  value=10: PASSED validation (result: 10)

============================================================

Mathematical check:
  5 > 5.5 = False
  6 > 5.5 = True
```
</details>

## Why This Is A Bug

The `gt(5.5)` constraint should reject any value `v` where `v > 5.5` evaluates to False. Since `5 > 5.5` is mathematically False in Python, the value 5 (and all values ≤ 5) should be rejected by validation. However, the constraint is not being applied at all when there's a type mismatch between the constraint value (float 5.5) and the field type (int).

This violates the documented behavior of the `gt()` method which states it "constrains a value to be greater than a certain value" without any qualification about type matching requirements. The behavior is also inconsistent with the other comparison operators:
- `ge(5.5)` correctly rejects 5 (as shown in the reproduction)
- `lt()` and `le()` also work correctly with type mismatches

This creates a critical data validation failure where invalid data silently passes through, potentially corrupting data or causing downstream logic errors. The API accepts the syntax `validate_as(int).gt(5.5)` without any error or warning, creating a reasonable expectation that it will validate correctly.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/experimental/pipeline.py`, specifically in the `_apply_constraint` function at lines 448-463.

The root cause is a structural difference in how `gt()` is implemented compared to the other comparison operators:

1. For `gt()`: The validation function `check_gt` is defined **inside** an `else` block (lines 458-463), meaning it only executes when the schema type doesn't exist or doesn't match the expected types.

2. For `ge()`, `lt()`, `le()`: The validation functions are defined **outside** the conditional blocks (lines 475-478, 490-493, 505-508), ensuring they always execute regardless of type matching.

When `gt(5.5)` is used on an integer field:
- Line 450 condition `s and s['type'] in {'int', 'float', 'decimal'}` is True
- Line 452 condition `s['type'] == 'int' and isinstance(gt, int)` is False (5.5 is a float)
- Lines 454 and 456 conditions also fail
- The code never enters the `else` block where `check_gt` is defined
- No validation is applied, and all values pass through

This is clearly an implementation error where the indentation/scoping of the validation function is incorrect.

## Proposed Fix

The fix is to unindent lines 459-463 to make the `check_gt` function definition and application consistent with the other operators:

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -455,12 +455,12 @@ def _apply_constraint(  # noqa: C901
                 s['gt'] = gt
             elif s['type'] == 'decimal' and isinstance(gt, Decimal):
                 s['gt'] = gt
-        else:

-            def check_gt(v: Any) -> bool:
-                return v > gt
+        def check_gt(v: Any) -> bool:
+            return v > gt

-            s = _check_func(check_gt, f'> {gt}', s)
+        s = _check_func(check_gt, f'> {gt}', s)
+
     elif isinstance(constraint, annotated_types.Ge):
         ge = constraint.ge
         if s and s['type'] in {'int', 'float', 'decimal'}:
```