# Bug Report: pydantic.experimental.pipeline not_in Constraint Uses Bitwise NOT Instead of Logical NOT

**Target**: `pydantic.experimental.pipeline._apply_constraint` (NotIn constraint)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `not_in` constraint is broken because it uses `operator.__not__` (bitwise NOT) instead of logical `not`. This causes the constraint to always pass validation, even for values that should be rejected.

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
```

**Failing input**: Any value in the forbidden set (e.g., `2`)

## Reproducing the Bug

```python
from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import transform

forbidden = {1, 2, 3}

class TestModel(BaseModel):
    field: int = transform(lambda x: x).not_in(forbidden)

try:
    model = TestModel(field=2)
    print(f"BUG: Value 2 was accepted: {model.field}")
    print("Expected: ValidationError")
except ValidationError as e:
    print(f"Correct: {e}")
```

**Expected output:**
```
Correct: ValidationError: [...]
```

**Actual output:**
```
BUG: Value 2 was accepted: 2
Expected: ValidationError
```

## Why This Is A Bug

At line 631, the code uses `operator.__not__` which is the bitwise NOT operator (`~`), not logical NOT:

```python
return operator.__not__(operator.__contains__(values, v))
```

In Python:
- `~True` (aka `~1`) → `-2` (truthy!)
- `~False` (aka `~0`) → `-1` (truthy!)

So the function ALWAYS returns a truthy value:
- If `v in values`: returns `-2` (truthy) → validation passes (WRONG!)
- If `v not in values`: returns `-1` (truthy) → validation passes (correct)

This means values that SHOULD be rejected are incorrectly accepted, completely breaking the constraint.

**Impact:**
- High severity: Silently accepts invalid data
- Security risk: If `not_in` is used to block dangerous inputs, they'll be accepted
- Violates documented API contract

## Fix

Replace `operator.__not__` with logical `not`:

```diff
diff --git a/pydantic/experimental/pipeline.py b/pydantic/experimental/pipeline.py
index 1234567..abcdefg 100644
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -628,7 +628,7 @@ def _apply_constraint(
         values = constraint.values

         def check_not_in(v: Any) -> bool:
-            return operator.__not__(operator.__contains__(values, v))
+            return not (v in values)

         s = _check_func(check_not_in, f'not in {values}', s)
```

Alternatively, use the negation directly:
```python
return v not in values
```