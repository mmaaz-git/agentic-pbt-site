# Bug Report: pydantic.experimental.pipeline Ge Constraint Inconsistency

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Ge` (greater-than-or-equal) constraint in `_apply_constraint` creates unnecessarily complex schemas compared to `Gt`, `Lt`, and `Le` constraints due to always applying an additional validator function even when the constraint can be set directly in the schema.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic.experimental.pipeline import _apply_constraint
from pydantic_core import core_schema as cs
import annotated_types


@given(st.integers(min_value=-1000, max_value=1000))
@settings(max_examples=200)
def test_ge_constraint_schema_structure_matches_gt(value):
    int_schema = cs.int_schema()

    ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(value))
    gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(value))

    assert ge_result['type'] == gt_result['type'], \
        f"Ge and Gt should produce same schema type, got {ge_result['type']} vs {gt_result['type']}"
```

**Failing input**: `value=0` (or any integer)

## Reproducing the Bug

```python
from pydantic.experimental.pipeline import _apply_constraint
from pydantic_core import core_schema as cs
import annotated_types

int_schema = cs.int_schema()

ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(5))
gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(5))

print(f"Ge(5) schema type: {ge_result['type']}")
print(f"Gt(5) schema type: {gt_result['type']}")
```

Output:
```
Ge(5) schema type: function-after
Gt(5) schema type: int
```

## Why This Is A Bug

The `Ge` constraint handler (lines 464-478 in pipeline.py) inconsistently applies a validator function compared to `Gt`, `Lt`, and `Le`:

- `Gt`, `Lt`, `Le`: Only add `_check_func` validator in the `else` branch when the constraint cannot be set directly
- `Ge`: Always adds `_check_func` validator, even after successfully setting the constraint in the schema

This creates unnecessarily complex schemas (wrapping with `function-after`) and inconsistent behavior between similar constraints.

## Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -471,11 +471,12 @@ def _apply_constraint(  # noqa: C901
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
+
     elif isinstance(constraint, annotated_types.Lt):
         lt = constraint.lt
         if s and s['type'] in {'int', 'float', 'decimal'}:
```