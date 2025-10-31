# Bug Report: pydantic.experimental.pipeline All Constraints and Transformations Silently Fail

**Target**: `pydantic.experimental.pipeline` (all constraints and transformations)
**Severity**: Critical
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

All constraints (`not_in`, `in_`, `eq`, `not_eq`, `gt`, `lt`, etc.) and transformations in `pydantic.experimental.pipeline` are completely non-functional. They silently accept all values without applying any validation or transformation, creating a severe data validation bypass.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import transform


@given(st.integers(min_value=0, max_value=100))
def test_not_in_constraint_rejects_forbidden_values(value):
    forbidden_values = {1, 2, 3, 5, 8, 13}

    class Model(BaseModel):
        field: int = transform(lambda x: x).not_in(forbidden_values)

    if value in forbidden_values:
        try:
            Model(field=value)
            assert False, f"Value {value} should be rejected but was accepted"
        except ValidationError:
            pass
    else:
        model = Model(field=value)
        assert model.field not in forbidden_values


if __name__ == "__main__":
    test_not_in_constraint_rejects_forbidden_values()
```

<details>

<summary>
**Failing input**: `value=1`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 25, in <module>
    test_not_in_constraint_rejects_forbidden_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 7, in test_not_in_constraint_rejects_forbidden_values
    def test_not_in_constraint_rejects_forbidden_values(value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 16, in test_not_in_constraint_rejects_forbidden_values
    assert False, f"Value {value} should be rejected but was accepted"
           ^^^^^
AssertionError: Value 1 should be rejected but was accepted
Falsifying example: test_not_in_constraint_rejects_forbidden_values(
    value=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/16/hypo.py:14
```
</details>

## Reproducing the Bug

```python
from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import transform

forbidden = {1, 2, 3}

class TestModel(BaseModel):
    field: int = transform(lambda x: x).not_in(forbidden)

print("Testing not_in constraint with forbidden values {1, 2, 3}:")
print()

# Test with a value that should be rejected (2 is in forbidden set)
print("Test 1: field=2 (should be rejected)")
try:
    model = TestModel(field=2)
    print(f"  BUG: Value 2 was accepted: {model.field}")
    print("  Expected: ValidationError")
except ValidationError as e:
    print(f"  Correct: ValidationError raised")
    print(f"  Error: {e}")

print()

# Test with a value that should be accepted (4 is not in forbidden set)
print("Test 2: field=4 (should be accepted)")
try:
    model = TestModel(field=4)
    print(f"  Correct: Value 4 was accepted: {model.field}")
except ValidationError as e:
    print(f"  BUG: Value 4 was rejected with error: {e}")
    print("  Expected: Value should be accepted")
```

<details>

<summary>
Output showing constraint bypass
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Testing not_in constraint with forbidden values {1, 2, 3}:

Test 1: field=2 (should be rejected)
  BUG: Value 2 was accepted: 2
  Expected: ValidationError

Test 2: field=4 (should be accepted)
  Correct: Value 4 was accepted: 4
```
</details>

## Why This Is A Bug

This violates the fundamental contract of data validation in Pydantic. The experimental pipeline module promises to apply constraints and transformations, but none of them work:

1. **All constraints fail silently**: Testing reveals that `not_in`, `in_`, `eq`, `not_eq`, `gt`, `lt`, and other constraints all accept every value without validation
2. **Transformations are not applied**: Functions passed to `transform()` are never executed (e.g., `transform(lambda x: x * 10)` returns unchanged values)
3. **The constraint checking functions return correct booleans**: The internal `check_not_in` function correctly returns `False` for forbidden values and `True` for allowed values
4. **Complete validation bypass**: Despite correct internal logic, the validation is never enforced at the model level

This creates a critical security vulnerability where developers believe their data is being validated but it's actually accepting any input.

## Relevant Context

Further investigation shows the scope of the problem:

1. **All constraints are affected**: Testing multiple constraint types shows universal failure:
   - `eq(5)` accepts all values, not just 5
   - `not_eq(5)` accepts all values, including 5
   - `in_({1,2,3})` accepts all values, not just 1, 2, or 3
   - `gt(5)` accepts all values, including those ≤ 5

2. **The constraint logic is correct but not applied**: Debug traces show that the constraint checking functions (like `check_not_in`) return the correct boolean values, but these checks are not being enforced during model validation.

3. **The module includes an experimental warning** but still provides documented APIs that developers may rely on for critical validation logic.

Documentation: https://docs.pydantic.dev/latest/api/experimental/#pydantic.experimental.pipeline

## Proposed Fix

The issue appears to be in how the experimental pipeline integrates with Pydantic's core validation system. The constraints are being constructed but not properly applied during validation. Investigation suggests the problem may be in `__get_pydantic_core_schema__` method or how the pipeline steps are being processed.

A thorough review of the integration between `_Pipeline.__get_pydantic_core_schema__` and Pydantic's core validation system is needed. The constraint functions themselves work correctly - they just aren't being invoked during validation.

Note: The initial bug report incorrectly identified the issue as `operator.__not__` using bitwise NOT. Testing confirms `operator.__not__` correctly performs logical NOT. The real issue is that constraint validation is completely bypassed.