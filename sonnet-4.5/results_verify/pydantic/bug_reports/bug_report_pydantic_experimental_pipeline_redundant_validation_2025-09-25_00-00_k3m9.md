# Bug Report: pydantic.experimental.pipeline Redundant Validation in Ge/Lt/Le Constraints

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Ge`, `Lt`, and `Le` constraint implementations in `pydantic.experimental.pipeline._apply_constraint` add redundant validator functions even when schema-level constraints are already set, unlike the `Gt` constraint which correctly uses an `else` clause to avoid redundancy. This causes double validation and reduced performance for these constraints on numeric types.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.experimental.pipeline import validate_as
from pydantic import BaseModel
from typing import Annotated

@given(st.integers())
def test_ge_constraint_not_redundant(x):
    """Property: Ge constraint should only validate once"""

    class ModelGe(BaseModel):
        value: Annotated[int, validate_as(int).ge(5)]

    schema = ModelGe.__pydantic_core_schema__

    has_schema_constraint = False
    has_validator_function = False

    def check_schema(s):
        nonlocal has_schema_constraint, has_validator_function
        if isinstance(s, dict):
            if s.get('type') == 'int' and 'ge' in s:
                has_schema_constraint = True
            if s.get('type') in ('no-info-after-validator-function', 'no-info-plain-validator-function'):
                has_validator_function = True
            for v in s.values():
                if isinstance(v, dict):
                    check_schema(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            check_schema(item)

    check_schema(schema)

    assert not (has_schema_constraint and has_validator_function), \
        "Both schema constraint and validator function present - redundant validation!"
```

**Failing input**: Any integer (the bug is structural, not input-dependent)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from pydantic.experimental.pipeline import validate_as
from pydantic import BaseModel
from typing import Annotated

class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(5)]

class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(5)]

print("Gt constraint (correct - no redundancy):")
print(ModelGt.__pydantic_core_schema__)
print("\nGe constraint (bug - has redundant validation):")
print(ModelGe.__pydantic_core_schema__)
```

## Why This Is A Bug

In `pydantic/experimental/pipeline.py`, the `_apply_constraint` function handles `Gt` correctly (lines 448-463):

```python
if isinstance(constraint, annotated_types.Gt):
    gt = constraint.gt
    if s and s['type'] in {'int', 'float', 'decimal'}:
        s = s.copy()
        if s['type'] == 'int' and isinstance(gt, int):
            s['gt'] = gt
        # ... (set schema constraint)
    else:  # <-- NOTE THE ELSE CLAUSE
        def check_gt(v: Any) -> bool:
            return v > gt
        s = _check_func(check_gt, f'> {gt}', s)
```

The `else` clause ensures that the validator function is only added when the schema constraint is NOT set.

However, `Ge` (lines 464-478), `Lt` (lines 479-493), and `Le` (lines 494-508) are missing this `else` clause:

```python
elif isinstance(constraint, annotated_types.Ge):
    ge = constraint.ge
    if s and s['type'] in {'int', 'float', 'decimal'}:
        s = s.copy()
        if s['type'] == 'int' and isinstance(ge, int):
            s['ge'] = ge
        # ... (set schema constraint)
    # MISSING else CLAUSE!
    def check_ge(v: Any) -> bool:
        return v >= ge
    s = _check_func(check_ge, f'>= {ge}', s)  # <-- ALWAYS EXECUTED
```

This causes the validator function to be added even when the schema already has the constraint, resulting in double validation.

## Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -472,10 +472,11 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(ge, Decimal):
                 s['ge'] = ge

-        def check_ge(v: Any) -> bool:
-            return v >= ge
+    else:
+            def check_ge(v: Any) -> bool:
+                return v >= ge

-        s = _check_func(check_ge, f'>= {ge}', s)
+            s = _check_func(check_ge, f'>= {ge}', s)
     elif isinstance(constraint, annotated_types.Lt):
         lt = constraint.lt
         if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -487,10 +488,11 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(lt, Decimal):
                 s['lt'] = lt

-        def check_lt(v: Any) -> bool:
-            return v < lt
+    else:
+            def check_lt(v: Any) -> bool:
+                return v < lt

-        s = _check_func(check_lt, f'< {lt}', s)
+            s = _check_func(check_lt, f'< {lt}', s)
     elif isinstance(constraint, annotated_types.Le):
         le = constraint.le
         if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -502,10 +504,11 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(le, Decimal):
                 s['le'] = le

-        def check_le(v: Any) -> bool:
-            return v <= le
+    else:
+            def check_le(v: Any) -> bool:
+                return v <= le

-        s = _check_func(check_le, f'<= {le}', s)
+            s = _check_func(check_le, f'<= {le}', s)
     elif isinstance(constraint, annotated_types.Len):
         min_len = constraint.min_length
         max_len = constraint.max_length