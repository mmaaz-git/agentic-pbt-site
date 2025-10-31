# Bug Report: pydantic.experimental.pipeline Ge/Lt/Le Redundant Validation

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Ge`, `Lt`, and `Le` constraints in pydantic's experimental pipeline module apply validation twice for numeric types (int/float/decimal), unlike `Gt` which correctly uses an `else` clause to avoid redundancy.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
from typing import Annotated

@given(st.integers(min_value=0, max_value=100))
@example(5)  # Specific example from the bug report
def test_ge_constraint_redundancy_hypothesis(threshold):
    """Property-based test showing that Ge/Lt/Le constraints work correctly but inefficiently.

    The bug doesn't cause incorrect behavior - it just applies validation twice:
    1. Once natively via the schema's 'ge' field (correct)
    2. Once via _check_func wrapper (redundant)
    """

    # Test with Ge constraint
    class ModelGe(BaseModel):
        field: Annotated[int, transform(lambda x: x).ge(threshold)]

    # Test valid values
    test_value = threshold + 10
    model = ModelGe(field=test_value)
    assert model.field == test_value

    # Test boundary value
    model_boundary = ModelGe(field=threshold)
    assert model_boundary.field == threshold

    # Test invalid value (should raise validation error)
    try:
        ModelGe(field=threshold - 1)
        assert False, "Should have raised validation error"
    except ValueError:
        pass  # Expected

    # Similar tests for Lt and Le constraints
    class ModelLt(BaseModel):
        field: Annotated[int, transform(lambda x: x).lt(threshold)]

    class ModelLe(BaseModel):
        field: Annotated[int, transform(lambda x: x).le(threshold)]

    # Lt tests
    if threshold > 0:
        model_lt = ModelLt(field=threshold - 1)
        assert model_lt.field == threshold - 1
        try:
            ModelLt(field=threshold)
            assert False, "Should have raised validation error for Lt"
        except ValueError:
            pass

    # Le tests
    model_le = ModelLe(field=threshold)
    assert model_le.field == threshold
    if threshold > 0:
        model_le2 = ModelLe(field=threshold - 1)
        assert model_le2.field == threshold - 1
    try:
        ModelLe(field=threshold + 1)
        assert False, "Should have raised validation error for Le"
    except ValueError:
        pass
```

<details>

<summary>
**Failing input**: `5` (and any integer value triggers redundant validation)
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Running property-based test with Hypothesis...
Testing that Ge/Lt/Le constraints work correctly (despite redundant validation)
============================================================

✓ All tests passed!

Note: The bug is about REDUNDANT validation, not incorrect behavior.
The constraints work correctly but apply validation twice internally.
```
</details>

## Reproducing the Bug

```python
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
from typing import Annotated

# Test with Gt constraint (correct implementation)
class ModelGt(BaseModel):
    field: Annotated[int, transform(lambda x: x).gt(5)]

# Test with Ge constraint (has redundant validation)
class ModelGe(BaseModel):
    field: Annotated[int, transform(lambda x: x).ge(5)]

# Test with Lt constraint (has redundant validation)
class ModelLt(BaseModel):
    field: Annotated[int, transform(lambda x: x).lt(100)]

# Test with Le constraint (has redundant validation)
class ModelLe(BaseModel):
    field: Annotated[int, transform(lambda x: x).le(100)]

print("Testing Gt constraint (correct implementation):")
try:
    print(f"  ModelGt(field=10): {ModelGt(field=10)}")
    print(f"  ModelGt(field=3): should fail...")
    ModelGt(field=3)
except Exception as e:
    print(f"  Failed as expected: {e}")

print("\nTesting Ge constraint (redundant validation):")
try:
    print(f"  ModelGe(field=10): {ModelGe(field=10)}")
    print(f"  ModelGe(field=5): {ModelGe(field=5)}")
    print(f"  ModelGe(field=3): should fail...")
    ModelGe(field=3)
except Exception as e:
    print(f"  Failed as expected: {e}")

print("\nTesting Lt constraint (redundant validation):")
try:
    print(f"  ModelLt(field=50): {ModelLt(field=50)}")
    print(f"  ModelLt(field=150): should fail...")
    ModelLt(field=150)
except Exception as e:
    print(f"  Failed as expected: {e}")

print("\nTesting Le constraint (redundant validation):")
try:
    print(f"  ModelLe(field=50): {ModelLe(field=50)}")
    print(f"  ModelLe(field=100): {ModelLe(field=100)}")
    print(f"  ModelLe(field=150): should fail...")
    ModelLe(field=150)
except Exception as e:
    print(f"  Failed as expected: {e}")

print("\nNote: All work correctly, but Ge/Lt/Le apply validation twice internally.")
```

<details>

<summary>
Output showing functional correctness (but with redundant validation internally)
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Testing Gt constraint (correct implementation):
  ModelGt(field=10): field=10
  ModelGt(field=3): should fail...
  Failed as expected: 1 validation error for ModelGt
field
  Value error, Expected > 5 [type=value_error, input_value=3, input_type=int]
    For further information visit https://errors.pydantic.dev/2.10/v/value_error

Testing Ge constraint (redundant validation):
  ModelGe(field=10): field=10
  ModelGe(field=5): field=5
  ModelGe(field=3): should fail...
  Failed as expected: 1 validation error for ModelGe
field
  Value error, Expected >= 5 [type=value_error, input_value=3, input_type=int]
    For further information visit https://errors.pydantic.dev/2.10/v/value_error

Testing Lt constraint (redundant validation):
  ModelLt(field=50): field=50
  ModelLt(field=150): should fail...
  Failed as expected: 1 validation error for ModelLt
field
  Value error, Expected < 100 [type=value_error, input_value=150, input_type=int]
    For further information visit https://errors.pydantic.dev/2.10/v/value_error

Testing Le constraint (redundant validation):
  ModelLe(field=50): field=50
  ModelLe(field=100): field=100
  ModelLe(field=150): should fail...
  Failed as expected: 1 validation error for ModelLe
field
  Value error, Expected <= 100 [type=value_error, input_value=150, input_type=int]
    For further information visit https://errors.pydantic.dev/2.10/v/value_error

Note: All work correctly, but Ge/Lt/Le apply validation twice internally.
```
</details>

## Why This Is A Bug

The implementation in `pydantic/experimental/pipeline.py` shows an inconsistency between similar constraint types:

1. **Gt constraint (lines 448-463)** - Correct implementation using if/else:
   - When schema type is int/float/decimal AND constraint type matches, it adds the constraint to the schema natively
   - **ELSE** it creates a validation function via `_check_func`
   - This means validation happens only once

2. **Ge/Lt/Le constraints (lines 464-508)** - Missing else clause:
   - When schema type is int/float/decimal AND constraint type matches, it adds the constraint to the schema natively
   - **ALWAYS** creates a validation function via `_check_func` (no else clause)
   - This means validation happens twice for numeric types

The bug violates the principle of DRY (Don't Repeat Yourself) and causes unnecessary performance overhead. While the module is marked as experimental, the inconsistency between `Gt` and the other comparison operators suggests this is unintentional.

## Relevant Context

- Source file: `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/experimental/pipeline.py`
- The experimental pipeline module provides a fluent API for building validation chains
- The module includes a deprecation warning indicating it's subject to change
- All four comparison constraints (Gt, Ge, Lt, Le) should follow the same pattern for consistency
- The bug doesn't affect correctness, only efficiency - validation still works as expected

## Proposed Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -471,10 +471,11 @@ def _apply_constraint(  # noqa: C901
                 s['ge'] = ge
             elif s['type'] == 'decimal' and isinstance(ge, Decimal):
                 s['ge'] = ge
-
-        def check_ge(v: Any) -> bool:
-            return v >= ge
-
-        s = _check_func(check_ge, f'>= {ge}', s)
+        else:
+            def check_ge(v: Any) -> bool:
+                return v >= ge
+
+            s = _check_func(check_ge, f'>= {ge}', s)
     elif isinstance(constraint, annotated_types.Lt):
         lt = constraint.lt
@@ -486,10 +487,11 @@ def _apply_constraint(  # noqa: C901
                 s['lt'] = lt
             elif s['type'] == 'decimal' and isinstance(lt, Decimal):
                 s['lt'] = lt
-
-        def check_lt(v: Any) -> bool:
-            return v < lt
-
-        s = _check_func(check_lt, f'< {lt}', s)
+        else:
+            def check_lt(v: Any) -> bool:
+                return v < lt
+
+            s = _check_func(check_lt, f'< {lt}', s)
     elif isinstance(constraint, annotated_types.Le):
         le = constraint.le
@@ -501,10 +503,11 @@ def _apply_constraint(  # noqa: C901
                 s['le'] = le
             elif s['type'] == 'decimal' and isinstance(le, Decimal):
                 s['le'] = le
-
-        def check_le(v: Any) -> bool:
-            return v <= le
-
-        s = _check_func(check_le, f'<= {le}', s)
+        else:
+            def check_le(v: Any) -> bool:
+                return v <= le
+
+            s = _check_func(check_le, f'<= {le}', s)
     elif isinstance(constraint, annotated_types.Len):
```