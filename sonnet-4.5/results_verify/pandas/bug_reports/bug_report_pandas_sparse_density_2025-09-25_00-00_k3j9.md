# Bug Report: pandas.core.sparse.SparseArray.density Division by Zero

**Target**: `pandas.core.sparse.SparseArray.density`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `density` property of `SparseArray` raises `ZeroDivisionError` when called on an empty array.

## Property-Based Test

```python
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st
import math

@st.composite
def sparse_arrays(draw, min_size=0, max_size=100):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    data = draw(st.lists(st.integers(), min_size=size, max_size=size))
    return SparseArray(data)

@given(sparse_arrays())
def test_density_property(arr):
    expected_density = arr.npoints / arr.sp_index.length if arr.sp_index.length > 0 else 0.0
    assert math.isclose(arr.density, expected_density)
```

**Failing input**: `SparseArray([])`

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

empty_arr = SparseArray([])
density = empty_arr.density
```

## Why This Is A Bug

The `density` property should return a valid floating-point value for empty arrays (likely 0.0), not crash with a division by zero error. This violates the expectation that properties should be accessible for all valid object states.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -705,7 +705,10 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         >>> s.density
         0.6
         """
-        return self.sp_index.npoints / self.sp_index.length
+        if self.sp_index.length == 0:
+            return 0.0
+        else:
+            return self.sp_index.npoints / self.sp_index.length
```