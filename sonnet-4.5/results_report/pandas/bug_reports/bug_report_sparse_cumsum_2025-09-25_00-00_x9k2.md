# Bug Report: SparseArray.cumsum() Infinite Recursion

**Target**: `pandas.core.arrays.sparse.array.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Calling `cumsum()` on a SparseArray with a non-null fill value causes infinite recursion, eventually leading to a RecursionError.

## Property-Based Test

```python
import numpy as np
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100))
def test_cumsum_matches_dense(data):
    arr = SparseArray(data, fill_value=0)

    if not arr._null_fill_value:
        dense = arr.to_dense()
        sparse_cumsum = arr.cumsum().to_dense()
        dense_cumsum = dense.cumsum()
        np.testing.assert_array_equal(sparse_cumsum, dense_cumsum)
```

**Failing input**: `SparseArray([1, 2, 3], fill_value=0)`

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

arr = SparseArray([1, 2, 3], fill_value=0)
result = arr.cumsum()
```

## Why This Is A Bug

The documentation states that `cumsum()` should return the cumulative sum of the array. However, when the fill value is non-null (e.g., 0 for integers), the method enters infinite recursion.

Looking at the code in `array.py:1549-1550`:
```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()
```

This line creates a new SparseArray from the dense representation. When no fill_value is specified in the SparseArray constructor, it infers the default fill value based on dtype (0 for integers). This means the newly created SparseArray also has `_null_fill_value = False`, causing the same code path to execute again, leading to infinite recursion.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,11 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            # Convert to dense, compute cumsum, then convert back with NaN fill_value
+            # to avoid infinite recursion
+            dense_cumsum = self.to_dense().cumsum()
+            # Use NaN as fill value to ensure _null_fill_value=True in the result
+            return SparseArray(dense_cumsum, fill_value=np.nan)

         return SparseArray(
             self.sp_values.cumsum(),
```