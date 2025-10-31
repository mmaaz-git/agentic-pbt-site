# Bug Report: pydantic.experimental.pipeline Ge/Le/Lt Double Validation

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Ge`, `Lt`, and `Le` constraints in pydantic's experimental pipeline API apply validation twice on numeric types (int/float/decimal), while `Gt` correctly applies validation only once. This inconsistency results in unnecessary performance overhead and code duplication.

## Property-Based Test

```python
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
from typing import Annotated
from hypothesis import given, strategies as st


class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(10)]


class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(10)]


def count_validators_in_schema(schema, path=""):
    count = 0
    if isinstance(schema, dict):
        if schema.get('type') == 'function-after' or schema.get('type') == 'no-info':
            count += 1
        for key, value in schema.items():
            count += count_validators_in_schema(value, f"{path}.{key}")
    elif isinstance(schema, (list, tuple)):
        for i, item in enumerate(schema):
            count += count_validators_in_schema(item, f"{path}[{i}]")
    return count


@given(st.integers(min_value=11))
def test_gt_ge_schema_consistency(x):
    gt_schema = ModelGt.__pydantic_core_schema__
    ge_schema = ModelGe.__pydantic_core_schema__

    gt_validators = count_validators_in_schema(gt_schema)
    ge_validators = count_validators_in_schema(ge_schema)

    assert gt_validators == ge_validators, \
        f"Gt has {gt_validators} validators but Ge has {ge_validators} validators"
```

**Failing input**: Any integer value (e.g., `x=15`)

## Reproducing the Bug

```python
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
from typing import Annotated


class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(10)]


class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(10)]


gt_schema = ModelGt.__pydantic_core_schema__
ge_schema = ModelGe.__pydantic_core_schema__

print("Gt schema:", gt_schema['schema']['fields']['value']['schema'])
print("Ge schema:", ge_schema['schema']['fields']['value']['schema'])
```

**Output:**
```
Gt schema: {'type': 'int', 'gt': 10}
Ge schema: {'function': {'type': 'no-info', 'function': <function>}, 'schema': {'type': 'int', 'ge': 10}, 'type': 'function-after'}
```

The Ge constraint has both the native `{'type': 'int', 'ge': 10}` constraint AND an additional function-after validator, resulting in double validation.

## Why This Is A Bug

Looking at the source code in `pydantic/experimental/pipeline.py`:

**Gt implementation (lines 448-463):**
```python
if isinstance(constraint, annotated_types.Gt):
    gt = constraint.gt
    if s and s['type'] in {'int', 'float', 'decimal'}:
        s = s.copy()
        if s['type'] == 'int' and isinstance(gt, int):
            s['gt'] = gt
        # ... more type checks
    else:  # <-- Note the else clause
        def check_gt(v: Any) -> bool:
            return v > gt
        s = _check_func(check_gt, f'> {gt}', s)
```

**Ge implementation (lines 464-478):**
```python
elif isinstance(constraint, annotated_types.Ge):
    ge = constraint.ge
    if s and s['type'] in {'int', 'float', 'decimal'}:
        s = s.copy()
        if s['type'] == 'int' and isinstance(ge, int):
            s['ge'] = ge
        # ... more type checks

    def check_ge(v: Any) -> bool:  # <-- No else clause!
        return v >= ge

    s = _check_func(check_ge, f'>= {ge}', s)
```

For **Gt**: The `_check_func` is only called in the `else` branch (when schema is not a matching numeric type).

For **Ge** (and Lt/Le): The `_check_func` is ALWAYS called, even when the schema constraint was already set.

This inconsistency causes Ge/Lt/Le to validate values twice:
1. First via the native pydantic-core constraint (`s['ge'] = ge`)
2. Then via the custom validator function (`_check_func`)

## Fix

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
+        else:
+            def check_ge(v: Any) -> bool:
+                return v >= ge
+
         s = _check_func(check_ge, f'>= {ge}', s)
     elif isinstance(constraint, annotated_types.Lt):
         lt = constraint.lt
@@ -487,10 +488,11 @@ def _apply_constraint(  # noqa: C901
                 s['lt'] = lt
             elif s['type'] == 'decimal' and isinstance(lt, Decimal):
                 s['lt'] = lt
-
-        def check_lt(v: Any) -> bool:
-            return v < lt
-
+        else:
+            def check_lt(v: Any) -> bool:
+                return v < lt
+
         s = _check_func(check_lt, f'< {lt}', s)
     elif isinstance(constraint, annotated_types.Le):
         le = constraint.le
@@ -503,10 +505,11 @@ def _apply_constraint(  # noqa: C901
                 s['le'] = le
             elif s['type'] == 'decimal' and isinstance(le, Decimal):
                 s['le'] = le
-
-        def check_le(v: Any) -> bool:
-            return v <= le
-
+        else:
+            def check_le(v: Any) -> bool:
+                return v <= le
+
         s = _check_func(check_le, f'<= {le}', s)
     elif isinstance(constraint, annotated_types.Len):
         min_len = constraint.min_length
```

The fix adds `else:` clauses to match the Gt implementation, ensuring that `_check_func` is only called when the native schema constraint cannot be used.