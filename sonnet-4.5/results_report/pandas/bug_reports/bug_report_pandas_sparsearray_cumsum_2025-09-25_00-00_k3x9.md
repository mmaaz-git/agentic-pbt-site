# Bug Report: pandas.core.sparse.SparseArray.cumsum Infinite Recursion

**Target**: `pandas.core.sparse.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method of `SparseArray` enters infinite recursion when called on arrays with non-null fill values, causing a RecursionError crash.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.arrays import SparseArray
import numpy as np

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50))
def test_cumsum_consistent_with_dense(data):
    sparse = SparseArray(data)
    dense = sparse.to_dense()

    sparse_cumsum = sparse.cumsum()
    dense_cumsum = np.cumsum(dense)

    assert np.array_equal(sparse_cumsum.to_dense(), dense_cumsum)
```

**Failing input**: `[1]` (or any list with non-zero integers)

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

sparse = SparseArray([1, 0, 0, 2])
result = sparse.cumsum()
```

Output:
```
RecursionError: maximum recursion depth exceeded
```

## Why This Is A Bug

The `cumsum()` method is documented to return the cumulative sum of the array. However, when the fill value is not null, the implementation creates infinite recursion by calling:

```python
return SparseArray(self.to_dense()).cumsum()
```

This creates a new `SparseArray` from the dense representation and calls `cumsum()` on it. The new `SparseArray` has the same fill value, so it also takes the same code path, leading to infinite recursion.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, ExtensionArray):
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return SparseArray(self.to_dense().cumsum())

         return SparseArray(
             self.sp_values.cumsum(),
```

The fix is to call `cumsum()` on the dense numpy array, not on a new `SparseArray`.