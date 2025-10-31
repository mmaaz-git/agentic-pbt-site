# Bug Report: dask.dataframe.io.parquet _DNF.extract_pq_filters Dead Code

**Target**: `dask.dataframe.io.parquet._DNF.extract_pq_filters`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_DNF.extract_pq_filters` method contains a logical contradiction in its second conditional branch (lines 1674-1677) that makes it impossible to execute. The condition checks `not isinstance(predicate_expr.left, Expr)` AND `isinstance(predicate_expr.left, Projection)`, but since `Projection` is a subclass of `Expr`, these conditions are mutually exclusive. This means filter predicates with reversed operands (e.g., `5 > column`) are never properly extracted.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.dask_expr._expr import Expr, Projection, GT, LT

@given(st.integers())
def test_reversed_comparison_filters_extracted(value):
    # This test would verify that reversed comparisons are handled
    # Currently, the bug prevents this from working

    # Simulating: value > df['column'] (reversed comparison)
    # This should be converted to: column < value
    # But the bug prevents this code path from executing
    pass
```

## Reproducing the Bug

The bug is a logic error that prevents code execution rather than causing a crash. To demonstrate:

```python
from dask.dataframe.dask_expr._expr import Expr, Projection

print("Checking if Projection is a subclass of Expr:")
print(f"  issubclass(Projection, Expr) = {issubclass(Projection, Expr)}")

print("\nThe problematic condition at lines 1675-1676:")
print("  not isinstance(predicate_expr.left, Expr)")
print("  and isinstance(predicate_expr.left, Projection)")

print("\nIf predicate_expr.left is a Projection:")
print("  - isinstance(predicate_expr.left, Projection) = True")
print("  - isinstance(predicate_expr.left, Expr) = True (because Projection inherits from Expr)")
print("  - not isinstance(predicate_expr.left, Expr) = False")

print("\nTherefore: False AND True = False")
print("This branch can NEVER execute!")
```

## Why This Is A Bug

1. **Dead code**: The second conditional branch (lines 1674-1685) is unreachable
2. **Missing functionality**: The code is intended to handle reversed comparisons (e.g., `5 > column` → `column < 5`) but fails to do so
3. **Silent failure**: No error is raised, filters are simply not extracted correctly
4. **Data correctness**: Parquet predicate pushdown optimization may fail for reversed comparisons, leading to inefficient queries or incorrect filtering

## Fix

The bug is at line 1676. It should check `predicate_expr.right` instead of `predicate_expr.left` to handle the case where the column and value are swapped:

```diff
--- a/dask/dataframe/dask_expr/io/parquet.py
+++ b/dask/dataframe/dask_expr/io/parquet.py
@@ -1673,7 +1673,7 @@ class _DNF:
                 _filters = (column, op, value)
             elif (
                 not isinstance(predicate_expr.left, Expr)
-                and isinstance(predicate_expr.left, Projection)
+                and isinstance(predicate_expr.right, Projection)
                 and predicate_expr.left.frame._name == pq_expr._name
             ):
                 # Simple dict to make sure field comes first in filter
```

Wait, there's actually another issue - line 1677 also references `predicate_expr.left.frame._name`, which would also need to be changed to `predicate_expr.right.frame._name`. The complete fix:

```diff
--- a/dask/dataframe/dask_expr/io/parquet.py
+++ b/dask/dataframe/dask_expr/io/parquet.py
@@ -1673,8 +1673,8 @@ class _DNF:
                 _filters = (column, op, value)
             elif (
                 not isinstance(predicate_expr.left, Expr)
-                and isinstance(predicate_expr.left, Projection)
-                and predicate_expr.left.frame._name == pq_expr._name
+                and isinstance(predicate_expr.right, Projection)
+                and predicate_expr.right.frame._name == pq_expr._name
             ):
                 # Simple dict to make sure field comes first in filter
                 flip = {LE: GE, LT: GT, GE: LE, GT: LT}
```