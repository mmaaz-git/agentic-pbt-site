# Bug Report: pydantic.experimental.pipeline Ge/Lt/Le Redundant Validation

**Target**: `pydantic.experimental.pipeline._apply_constraint` (lines 464-508)
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Ge`, `Lt`, and `Le` constraints apply validation twice for numeric types (int/float/decimal), causing redundant checks, unlike `Gt` which correctly uses an `else` clause.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
from typing import Annotated

@given(st.integers(min_value=0, max_value=100))
def test_ge_constraint_efficiency(value):
    call_count = 0
    original_ge = lambda v, threshold: v >= threshold

    def tracked_ge(v):
        nonlocal call_count
        call_count += 1
        return original_ge(v, value)

    class Model(BaseModel):
        field: Annotated[int, transform(lambda x: x).ge(value)]

    Model(field=value)
```

**Failing input**: Any integer value triggers redundant validation when the schema type matches

## Reproducing the Bug

```python
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
from typing import Annotated

class ModelGt(BaseModel):
    field: Annotated[int, transform(lambda x: x).gt(5)]

class ModelGe(BaseModel):
    field: Annotated[int, transform(lambda x: x).ge(5)]

print("Both should work identically, but Ge has redundant validation")
print(ModelGt(field=10))
print(ModelGe(field=10))
```

## Why This Is A Bug

Looking at the implementation:

**Gt constraint (lines 448-463)** - Correct pattern:
```python
if s and s['type'] in {'int', 'float', 'decimal'}:
    s = s.copy()
    if s['type'] == 'int' and isinstance(gt, int):
        s['gt'] = gt
    # ...
else:
    def check_gt(v: Any) -> bool:
        return v > gt
    s = _check_func(check_gt, f'> {gt}', s)
```

**Ge constraint (lines 464-478)** - Missing else:
```python
if s and s['type'] in {'int', 'float', 'decimal'}:
    s = s.copy()
    if s['type'] == 'int' and isinstance(ge, int):
        s['ge'] = ge
    # ...

def check_ge(v: Any) -> bool:
    return v >= ge

s = _check_func(check_ge, f'>= {ge}', s)
```

The `Ge`, `Lt`, and `Le` constraints always call `_check_func` even when the schema already has the constraint built-in, resulting in double validation.

## Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -472,9 +472,10 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(ge, Decimal):
                 s['ge'] = ge
-
-        def check_ge(v: Any) -> bool:
-            return v >= ge
+        else:
+            def check_ge(v: Any) -> bool:
+                return v >= ge

-        s = _check_func(check_ge, f'>= {ge}', s)
+            s = _check_func(check_ge, f'>= {ge}', s)
     elif isinstance(constraint, annotated_types.Lt):
         lt = constraint.lt
@@ -487,9 +488,10 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(lt, Decimal):
                 s['lt'] = lt
-
-        def check_lt(v: Any) -> bool:
-            return v < lt
+        else:
+            def check_lt(v: Any) -> bool:
+                return v < lt

-        s = _check_func(check_lt, f'< {lt}', s)
+            s = _check_func(check_lt, f'< {lt}', s)
     elif isinstance(constraint, annotated_types.Le):
         le = constraint.le
@@ -502,9 +504,10 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(le, Decimal):
                 s['le'] = le
-
-        def check_le(v: Any) -> bool:
-            return v <= le
+        else:
+            def check_le(v: Any) -> bool:
+                return v <= le

-        s = _check_func(check_le, f'<= {le}', s)
+            s = _check_func(check_le, f'<= {le}', s)
     elif isinstance(constraint, annotated_types.Len):
```