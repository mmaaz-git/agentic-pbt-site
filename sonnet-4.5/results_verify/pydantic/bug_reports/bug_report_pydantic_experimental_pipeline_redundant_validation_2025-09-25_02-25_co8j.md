# Bug Report: pydantic.experimental.pipeline Redundant Validation in Ge, Lt, Le Constraints

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Ge`, `Lt`, and `Le` constraints apply validation twice: once through the optimized schema constraint and once through a redundant `function-after` validator. The `Gt` constraint correctly avoids this redundancy by only using `_check_func` when the schema optimization is not applied.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
from typing import Annotated

@given(st.integers(min_value=-1000, max_value=1000))
def test_ge_has_redundant_validation(boundary):
    class ModelGe(BaseModel):
        x: Annotated[int, validate_as(int).ge(boundary)]

    class ModelGt(BaseModel):
        x: Annotated[int, validate_as(int).gt(boundary)]

    ge_schema = ModelGe.__pydantic_core_schema__['schema']['fields']['x']['schema']
    gt_schema = ModelGt.__pydantic_core_schema__['schema']['fields']['x']['schema']

    assert gt_schema['type'] == 'int', "Gt has simple int schema (no redundant validation)"
    assert ge_schema['type'] == 'function-after', "Ge has redundant function-after wrapper"
    assert ge_schema['schema']['type'] == 'int', "Ge has int schema nested inside"
    assert 'ge' in ge_schema['schema'], "Ge has 'ge' constraint in nested schema"
```

**Failing input**: Any integer (the test demonstrates the bug exists for all boundaries)

## Reproducing the Bug

```python
import warnings
warnings.filterwarnings('ignore')

from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
from typing import Annotated

class ModelGt(BaseModel):
    x: Annotated[int, validate_as(int).gt(5)]

class ModelGe(BaseModel):
    x: Annotated[int, validate_as(int).ge(5)]

gt_schema = ModelGt.__pydantic_core_schema__['schema']['fields']['x']['schema']
ge_schema = ModelGe.__pydantic_core_schema__['schema']['fields']['x']['schema']

print("Gt schema (CORRECT):")
print(f"  {gt_schema}")

print("\nGe schema (BUG - has redundant validation):")
print(f"  {ge_schema}")

assert gt_schema == {'type': 'int', 'gt': 5}
assert ge_schema['type'] == 'function-after'
assert ge_schema['schema'] == {'type': 'int', 'ge': 5}
```

## Why This Is A Bug

The `Ge`, `Lt`, and `Le` constraints perform the same validation twice:
1. First through the optimized pydantic-core schema constraint (e.g., `{'type': 'int', 'ge': 5}`)
2. Then through a redundant `function-after` validator that checks the same condition

This causes:
- **Performance degradation**: Each validation is performed twice
- **Inconsistency**: The `Gt` constraint correctly avoids this redundancy
- **Code duplication**: The same logic is executed in two different ways

Looking at the source code in `pydantic/experimental/pipeline.py` around line 563, the `Gt` constraint has the correct pattern:

```python
if s and s['type'] in {'int', 'float', 'decimal'}:
    # Apply schema optimization
else:
    # Only use _check_func when schema optimization wasn't possible
    s = _check_func(check_gt, f'> {gt}', s)
```

But `Ge`, `Lt`, and `Le` always call `_check_func`, even after applying the schema optimization.

## Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -574,11 +574,11 @@ def _apply_constraint(  # noqa: C901
                 s['ge'] = ge
             elif s['type'] == 'decimal' and isinstance(ge, Decimal):
                 s['ge'] = ge
+    else:
+        def check_ge(v: Any) -> bool:
+            return v >= ge

-    def check_ge(v: Any) -> bool:
-        return v >= ge
-
-    s = _check_func(check_ge, f'>= {ge}', s)
+        s = _check_func(check_ge, f'>= {ge}', s)
 elif isinstance(constraint, annotated_types.Lt):
     lt = constraint.lt
     if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -589,11 +589,11 @@ def _apply_constraint(  # noqa: C901
             s['lt'] = lt
         elif s['type'] == 'decimal' and isinstance(lt, Decimal):
             s['lt'] = lt
+    else:
+        def check_lt(v: Any) -> bool:
+            return v < lt

-    def check_lt(v: Any) -> bool:
-        return v < lt
-
-    s = _check_func(check_lt, f'< {lt}', s)
+        s = _check_func(check_lt, f'< {lt}', s)
 elif isinstance(constraint, annotated_types.Le):
     le = constraint.le
     if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -604,11 +604,11 @@ def _apply_constraint(  # noqa: C901
             s['le'] = le
         elif s['type'] == 'decimal' and isinstance(le, Decimal):
             s['le'] = le
+    else:
+        def check_le(v: Any) -> bool:
+            return v <= le

-    def check_le(v: Any) -> bool:
-        return v <= le
-
-    s = _check_func(check_le, f'<= {le}', s)
+        s = _check_func(check_le, f'<= {le}', s)
 elif isinstance(constraint, annotated_types.Len):
     min_len = constraint.min_length
     max_len = constraint.max_length
```