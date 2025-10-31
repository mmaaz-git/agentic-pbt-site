# Bug Report: pydantic.experimental.pipeline Ge/Lt/Le Double Validation

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Ge` (>=), `Lt` (<), and `Le` (<=) constraint implementations apply validation twice when the schema type is int/float/decimal, unlike `Gt` (>) which correctly applies validation once.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from typing import Annotated
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as

def count_validators(schema):
    if schema.get('type') == 'function-after':
        return 1 + count_validators(schema.get('schema', {}))
    if schema.get('type') == 'chain':
        return sum(count_validators(s) for s in schema.get('steps', []))
    if 'fields' in schema:
        for field in schema['fields'].values():
            if 'schema' in field:
                return count_validators(field['schema'])
    if 'schema' in schema:
        return count_validators(schema['schema'])
    return 0

@given(value=st.integers())
def test_constraint_validators_consistency(value):
    class ModelGt(BaseModel):
        v: Annotated[int, validate_as(int).gt(value)]

    class ModelGe(BaseModel):
        v: Annotated[int, validate_as(int).ge(value)]

    gt_validators = count_validators(ModelGt.__pydantic_core_schema__)
    ge_validators = count_validators(ModelGe.__pydantic_core_schema__)

    assert gt_validators == ge_validators, \
        f"Gt has {gt_validators} validators but Ge has {ge_validators}"
```

**Failing input**: Any integer value

## Reproducing the Bug

```python
from typing import Annotated
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as

class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(5)]

class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(5)]

def count_validators(schema, depth=0):
    if schema.get('type') == 'function-after':
        return 1 + count_validators(schema.get('schema', {}), depth+1)
    if schema.get('type') == 'chain':
        return sum(count_validators(s, depth+1) for s in schema.get('steps', []))
    if 'fields' in schema:
        for field in schema['fields'].values():
            if 'schema' in field:
                return count_validators(field['schema'], depth+1)
    if 'schema' in schema:
        return count_validators(schema['schema'], depth+1)
    return 0

gt_validators = count_validators(ModelGt.__pydantic_core_schema__)
ge_validators = count_validators(ModelGe.__pydantic_core_schema__)

print(f"Gt constraint: {gt_validators} validator(s)")
print(f"Ge constraint: {ge_validators} validator(s)")
```

Output:
```
Gt constraint: 0 validator(s)
Ge constraint: 1 validator(s)
```

The Ge constraint has an extra validator function because the constraint is being checked twice.

## Why This Is A Bug

Looking at lines 448-508 of `pipeline.py`:

**Gt (CORRECT - lines 448-463)**:
```python
if isinstance(constraint, annotated_types.Gt):
    gt = constraint.gt
    if s and s['type'] in {'int', 'float', 'decimal'}:
        s = s.copy()
        if s['type'] == 'int' and isinstance(gt, int):
            s['gt'] = gt
        # ... similar for float and decimal
    else:  # ← NOTE: has else clause
        def check_gt(v: Any) -> bool:
            return v > gt
        s = _check_func(check_gt, f'> {gt}', s)
```

**Ge (BUGGY - lines 464-478)**:
```python
elif isinstance(constraint, annotated_types.Ge):
    ge = constraint.ge
    if s and s['type'] in {'int', 'float', 'decimal'}:
        s = s.copy()
        if s['type'] == 'int' and isinstance(ge, int):
            s['ge'] = ge
        # ... similar for float and decimal

    # ← NOTE: NO else clause - function always executes!
    def check_ge(v: Any) -> bool:
        return v >= ge

    s = _check_func(check_ge, f'>= {ge}', s)
```

**Impact**:
- When using Ge/Lt/Le with int/float/decimal types, the constraint is validated twice:
  1. Once by the schema's built-in constraint field (e.g., `s['ge'] = ge`)
  2. Once by the after-validator function
- This is inefficient and inconsistent with Gt behavior
- Same bug exists for Lt (lines 479-493) and Le (lines 494-508)

## Fix

Add `else` clauses to Ge, Lt, and Le constraint handlers, matching the pattern used in Gt:

```diff
 elif isinstance(constraint, annotated_types.Ge):
     ge = constraint.ge
     if s and s['type'] in {'int', 'float', 'decimal'}:
         s = s.copy()
         if s['type'] == 'int' and isinstance(ge, int):
             s['ge'] = ge
         elif s['type'] == 'float' and isinstance(ge, float):
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
```

Apply the same fix to Lt (lines 479-493) and Le (lines 494-508).