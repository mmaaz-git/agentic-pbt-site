# Bug Report: pydantic.experimental.pipeline Multiple Constraints Applied Twice

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Five constraint types (Ge, Lt, Le, Len, MultipleOf) in `_apply_constraint` are incorrectly applied twice: once as a schema constraint and again as a wrapper validator function. This is due to missing `else:` clauses that would prevent the double application. The `Gt` constraint is implemented correctly and shows the intended behavior.

## Property-Based Test

```python
import annotated_types
import pytest
from hypothesis import given, strategies as st, settings
from pydantic_core import core_schema as cs
from pydantic.experimental.pipeline import _apply_constraint


@settings(max_examples=500)
@given(st.integers(min_value=-1000, max_value=1000))
def test_ge_constraint_schema_structure(value):
    """
    Property: Constraint application should be consistent across similar constraint types.

    Gt (correctly) applies constraint once in schema only.
    Ge (bug) applies constraint in schema AND wraps with validator function.
    """
    int_schema = cs.int_schema()

    gt_schema = _apply_constraint(int_schema.copy(), annotated_types.Gt(value))
    ge_schema = _apply_constraint(int_schema.copy(), annotated_types.Ge(value))

    assert gt_schema['type'] == 'int'

    if ge_schema['type'] != gt_schema['type']:
        pytest.fail(f"Ge produces {ge_schema['type']} but Gt produces {gt_schema['type']}")
```

**Failing input**: Any integer value (e.g., `value=0`)

## Reproducing the Bug

```python
import annotated_types
from pydantic_core import core_schema as cs
from pydantic.experimental.pipeline import _apply_constraint

int_schema = cs.int_schema()

gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(10))
print(f"Gt schema: {gt_result}")

ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(10))
print(f"Ge schema: {ge_result}")
```

**Output:**
```
Gt schema: {'type': 'int', 'gt': 10}
Ge schema: {'type': 'function-after', 'function': {...}, 'schema': {'type': 'int', 'ge': 10}}
```

The Ge constraint produces a `function-after` wrapper even though the constraint is already in the schema.

## Why This Is A Bug

1. **Inefficiency**: The constraint is validated twice - once by pydantic-core's built-in schema validation and again by the wrapper function.

2. **Inconsistency**: Five similar constraint types behave differently from `Gt`, which is the correct implementation.

3. **Code Intent**: The code clearly intends to optimize by setting schema constraints when possible (as evidenced by the conditional logic), but the missing `else:` clauses defeat this optimization.

4. **Comparison with Gt**: The `Gt` constraint (lines 448-463) shows the correct pattern with an `else:` clause at line 458, while the buggy constraints lack this.

## Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -471,8 +471,9 @@ def _apply_constraint(
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
@@ -486,8 +487,9 @@ def _apply_constraint(
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
@@ -501,8 +503,9 @@ def _apply_constraint(
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
@@ -524,12 +527,13 @@ def _apply_constraint(
             s['min_length'] = min_len
         if max_len is not None:
             s['max_length'] = max_len
+    else:
+        def check_len(v: Any) -> bool:
+            if max_len is not None:
+                return (min_len <= len(v)) and (len(v) <= max_len)
+            return min_len <= len(v)

-    def check_len(v: Any) -> bool:
-        if max_len is not None:
-            return (min_len <= len(v)) and (len(v) <= max_len)
-        return min_len <= len(v)
-
-    s = _check_func(check_len, f'length >= {min_len} and length <= {max_len}', s)
+        s = _check_func(check_len, f'length >= {min_len} and length <= {max_len}', s)
 elif isinstance(constraint, annotated_types.MultipleOf):
     multiple_of = constraint.multiple_of
     if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -542,8 +546,9 @@ def _apply_constraint(
             s['multiple_of'] = multiple_of
         elif s['type'] == 'decimal' and isinstance(multiple_of, Decimal):
             s['multiple_of'] = multiple_of
+    else:
+        def check_multiple_of(v: Any) -> bool:
+            return v % multiple_of == 0

-    def check_multiple_of(v: Any) -> bool:
-        return v % multiple_of == 0
-
-    s = _check_func(check_multiple_of, f'% {multiple_of} == 0', s)
+        s = _check_func(check_multiple_of, f'% {multiple_of} == 0', s)
```

The fix adds `else:` clauses and proper indentation to match the `Gt` constraint's correct implementation, ensuring constraints are only applied once when the schema type supports them natively.