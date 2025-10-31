# Bug Report: pandas.core.sparse.SparseArray cumsum() Infinite Recursion

**Target**: `pandas.core.arrays.sparse.array.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method on SparseArray causes infinite recursion when the fill value is not a null value (e.g., 0 for integers). This happens because the method recursively calls itself without a proper base case.

## Property-Based Test

```python
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st
import numpy as np

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=50))
def test_sparse_array_cumsum_matches_numpy(values):
    arr = SparseArray(values)
    sparse_cumsum = arr.cumsum()
    numpy_cumsum = np.cumsum(values)
    assert np.array_equal(sparse_cumsum.to_dense(), numpy_cumsum), \
        f"cumsum mismatch: {sparse_cumsum.to_dense()} != {numpy_cumsum}"
```

**Failing input**: `[1, 2]` (or any integer list)

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

arr = SparseArray([1, 0, 2], fill_value=0)
result = arr.cumsum()
```

Expected: `SparseArray([1, 1, 3], fill_value=...)`
Actual: `RecursionError: maximum recursion depth exceeded`

## Why This Is A Bug

The `cumsum()` method is documented to return a SparseArray with cumulative sums. However, when `_null_fill_value` is False (which happens when fill_value is 0, False, or any other non-null value), the implementation calls:

```python
return SparseArray(self.to_dense()).cumsum()
```

This creates a new SparseArray from the dense array, which will also have `_null_fill_value=False`, causing it to call `cumsum()` again in an infinite loop.

The issue is that the default fill value for integer SparseArrays is 0, which is not considered a "null" value, so almost all integer SparseArrays will trigger this bug.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         if axis is not None and axis >= self.ndim:
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return SparseArray(self.to_dense().cumsum())

         return SparseArray(
             self.sp_values.cumsum(),
```

The fix is simple: call `cumsum()` on the dense array before wrapping it in a SparseArray, rather than calling `cumsum()` on the SparseArray itself.