# Bug Report: pydantic.experimental.pipeline Gt Constraint Fails with Type Mismatch

**Target**: `pydantic.experimental.pipeline._apply_constraint` (Gt constraint handling)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `gt()` (greater than) constraint in pydantic's experimental pipeline API fails to validate when the constraint value type doesn't match the schema type (e.g., using `gt(5.5)` on an integer field). The constraint silently passes invalid values instead of raising a ValidationError.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import validate_as
from typing import Annotated
import pytest


@given(st.integers(min_value=-1000, max_value=5))
def test_gt_float_constraint_on_int(value):
    class Model(BaseModel):
        field: Annotated[int, validate_as(int).gt(5.5)]

    with pytest.raises(ValidationError):
        Model(field=value)


@given(st.integers(min_value=6, max_value=1000))
def test_gt_float_constraint_accepts_valid(value):
    class Model(BaseModel):
        field: Annotated[int, validate_as(int).gt(5.5)]

    m = Model(field=value)
    assert m.field == value
```

**Failing input**: Any integer ≤ 5 (e.g., `value=5`)

## Reproducing the Bug

```python
from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import validate_as
from typing import Annotated


class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(5.5)]


try:
    m = ModelGt(value=5)
    print(f"BUG: value=5 passed gt(5.5) validation! Result: {m.value}")
except ValidationError as e:
    print(f"Expected behavior: {e}")


class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(5.5)]


try:
    m = ModelGe(value=5)
    print(f"BUG: value=5 passed ge(5.5) validation!")
except ValidationError as e:
    print(f"Correct: ge(5.5) properly rejects 5")
```

**Output:**
```
BUG: value=5 passed gt(5.5) validation! Result: 5
Correct: ge(5.5) properly rejects 5
```

## Why This Is A Bug

The `gt(5.5)` constraint should reject the value `5` because `5 > 5.5` is False. However, the constraint is not being applied at all. This violates the documented behavior of the `gt()` method and creates a silent data validation failure.

The bug only affects the `gt()` constraint. The `ge()`, `lt()`, and `le()` constraints work correctly in the same scenario.

## Fix

The root cause is in `pipeline.py` lines 448-463. The `_check_func` call is inside an `else` block that only executes when the schema type doesn't match the expected types. When the schema is an int type but the constraint value is a float (type mismatch), neither the schema optimization NOR the fallback validation is applied.

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -450,13 +450,13 @@ def _apply_constraint(  # noqa: C901
         if s and s['type'] in {'int', 'float', 'decimal'}:
             s = s.copy()
             if s['type'] == 'int' and isinstance(gt, int):
                 s['gt'] = gt
             elif s['type'] == 'float' and isinstance(gt, float):
                 s['gt'] = gt
             elif s['type'] == 'decimal' and isinstance(gt, Decimal):
                 s['gt'] = gt
-    else:

-        def check_gt(v: Any) -> bool:
-            return v > gt
+    def check_gt(v: Any) -> bool:
+        return v > gt

-        s = _check_func(check_gt, f'> {gt}', s)
+    s = _check_func(check_gt, f'> {gt}', s)
     elif isinstance(constraint, annotated_types.Ge):
```

This change makes the `Gt` constraint behave consistently with `Ge`, `Lt`, and `Le` constraints (lines 464-508), which all correctly apply the validation function regardless of type matching.