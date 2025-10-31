# Bug Report: pydantic.experimental.pipeline Multiple Constraint Implementation Bugs

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Five constraint types (`Ge`, `Lt`, `Le`, `MultipleOf`, `Len`) in `_apply_constraint` create unnecessarily complex schemas by always applying an additional validator function even when the constraint can be set directly in the schema. Only `Gt` is implemented correctly with the proper `else` clause.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic.experimental.pipeline import _apply_constraint
from pydantic_core import core_schema as cs
import annotated_types


@given(st.integers(min_value=-100, max_value=100))
@settings(max_examples=200)
def test_ge_matches_gt_schema_structure(value):
    int_schema = cs.int_schema()
    ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(value))
    gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(value))
    assert ge_result['type'] == gt_result['type']


@given(st.integers(min_value=-100, max_value=100))
@settings(max_examples=200)
def test_lt_matches_gt_schema_structure(value):
    int_schema = cs.int_schema()
    lt_result = _apply_constraint(int_schema.copy(), annotated_types.Lt(value))
    gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(value))
    assert lt_result['type'] == gt_result['type']


@given(st.integers(min_value=-100, max_value=100))
@settings(max_examples=200)
def test_le_matches_gt_schema_structure(value):
    int_schema = cs.int_schema()
    le_result = _apply_constraint(int_schema.copy(), annotated_types.Le(value))
    gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(value))
    assert le_result['type'] == gt_result['type']


@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=200)
def test_multiple_of_matches_gt_schema_structure(value):
    int_schema = cs.int_schema()
    mult_result = _apply_constraint(int_schema.copy(), annotated_types.MultipleOf(value))
    gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(0))
    assert mult_result['type'] == gt_result['type']
```

**Failing input**: Any valid integer value for all tests

## Reproducing the Bug

```python
from pydantic.experimental.pipeline import _apply_constraint
from pydantic_core import core_schema as cs
import annotated_types

int_schema = cs.int_schema()

constraints = [
    ('Gt(5)', annotated_types.Gt(5)),
    ('Ge(5)', annotated_types.Ge(5)),
    ('Lt(10)', annotated_types.Lt(10)),
    ('Le(10)', annotated_types.Le(10)),
    ('MultipleOf(3)', annotated_types.MultipleOf(3)),
]

for name, constraint in constraints:
    result = _apply_constraint(int_schema.copy(), constraint)
    print(f"{name:20} -> type: {result['type']}")
```

Output:
```
Gt(5)                -> type: int
Ge(5)                -> type: function-after
Lt(10)               -> type: function-after
Le(10)               -> type: function-after
MultipleOf(3)        -> type: function-after
```

## Why This Is A Bug

In `_apply_constraint` (lines 448-548 in pipeline.py), only `Gt` is implemented correctly:

**Correct implementation (Gt, lines 448-463):**
```python
if s and s['type'] in {'int', 'float', 'decimal'}:
    s = s.copy()
    if s['type'] == 'int' and isinstance(gt, int):
        s['gt'] = gt
    # ...
else:  # ← This else clause is KEY
    def check_gt(v: Any) -> bool:
        return v > gt
    s = _check_func(check_gt, f'> {gt}', s)
```

**Buggy implementations (Ge, Lt, Le, MultipleOf, Len):**
- Missing the `else` clause before calling `_check_func`
- Always wrap with validator function even when constraint is set directly in schema
- Creates unnecessarily complex schemas (`function-after` wrapper)
- For `MultipleOf` with floats, the naive modulo check (`v % multiple_of == 0`) also introduces precision issues

This affects:
- `Ge` (lines 464-478)
- `Lt` (lines 479-493)
- `Le` (lines 494-508)
- `Len` (lines 509-533)
- `MultipleOf` (lines 534-548)

## Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -471,11 +471,12 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(ge, Decimal):
                 s['ge'] = ge
-
-    def check_ge(v: Any) -> bool:
-        return v >= ge
-
-    s = _check_func(check_ge, f'>= {ge}', s)
+    else:
+        def check_ge(v: Any) -> bool:
+            return v >= ge
+
+        s = _check_func(check_ge, f'>= {ge}', s)
+
     elif isinstance(constraint, annotated_types.Lt):
         lt = constraint.lt
         if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -487,11 +488,12 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(lt, Decimal):
                 s['lt'] = lt
-
-    def check_lt(v: Any) -> bool:
-        return v < lt
-
-    s = _check_func(check_lt, f'< {lt}', s)
+    else:
+        def check_lt(v: Any) -> bool:
+            return v < lt
+
+        s = _check_func(check_lt, f'< {lt}', s)
+
     elif isinstance(constraint, annotated_types.Le):
         le = constraint.le
         if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -503,11 +505,12 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(le, Decimal):
                 s['le'] = le
-
-    def check_le(v: Any) -> bool:
-        return v <= le
-
-    s = _check_func(check_le, f'<= {le}', s)
+    else:
+        def check_le(v: Any) -> bool:
+            return v <= le
+
+        s = _check_func(check_le, f'<= {le}', s)
+
     elif isinstance(constraint, annotated_types.Len):
         min_len = constraint.min_length
         max_len = constraint.max_length
@@ -525,11 +528,12 @@ def _apply_constraint(  # noqa: C901
                 s['min_length'] = min_len
             if max_len is not None:
                 s['max_length'] = max_len
-
-    def check_len(v: Any) -> bool:
-        if max_len is not None:
-            return (min_len <= len(v)) and (len(v) <= max_len)
-        return min_len <= len(v)
-
-    s = _check_func(check_len, f'length >= {min_len} and length <= {max_len}', s)
+    else:
+        def check_len(v: Any) -> bool:
+            if max_len is not None:
+                return (min_len <= len(v)) and (len(v) <= max_len)
+            return min_len <= len(v)
+
+        s = _check_func(check_len, f'length >= {min_len} and length <= {max_len}', s)
+
     elif isinstance(constraint, annotated_types.MultipleOf):
         multiple_of = constraint.multiple_of
         if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -542,11 +546,12 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(multiple_of, Decimal):
                 s['multiple_of'] = multiple_of
-
-    def check_multiple_of(v: Any) -> bool:
-        return v % multiple_of == 0
-
-    s = _check_func(check_multiple_of, f'% {multiple_of} == 0', s)
+    else:
+        def check_multiple_of(v: Any) -> bool:
+            return v % multiple_of == 0
+
+        s = _check_func(check_multiple_of, f'% {multiple_of} == 0', s)
+
     elif isinstance(constraint, annotated_types.Timezone):
         tz = constraint.tz
```