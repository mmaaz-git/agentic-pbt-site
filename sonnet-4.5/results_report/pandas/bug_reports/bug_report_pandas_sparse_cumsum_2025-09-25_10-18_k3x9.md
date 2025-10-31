# Bug Report: pandas.core.sparse.SparseArray.cumsum Infinite Recursion

**Target**: `pandas.core.arrays.sparse.array.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method on `SparseArray` enters infinite recursion when the fill value is not null (e.g., integer arrays with fill_value=0), causing a `RecursionError` instead of computing the cumulative sum.

## Property-Based Test

```python
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50)
)
@settings(max_examples=500)
def test_cumsum_length(data):
    arr = SparseArray(data)
    cumsum_result = arr.cumsum()
    assert len(cumsum_result) == len(arr)
```

**Failing input**: `[0]` (or any list of integers)

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

arr = SparseArray([1, 2, 3], fill_value=0)
result = arr.cumsum()
```

This code immediately enters infinite recursion and raises `RecursionError: maximum recursion depth exceeded`.

The bug occurs with any integer SparseArray (default fill_value=0), but works correctly with float arrays that have fill_value=NaN.

```python
from pandas.arrays import SparseArray
import numpy as np

arr_int = SparseArray([1, 2, 3])
arr_float = SparseArray([1.0, 2.0, 3.0])

arr_int.cumsum()
arr_float.cumsum()
```

## Why This Is A Bug

The `cumsum()` method is documented to return a cumulative sum, similar to numpy's `cumsum()`. Users expect `SparseArray([1, 2, 3]).cumsum()` to return a SparseArray equivalent to `[1, 3, 6]`, not to crash with infinite recursion.

The root cause is in line 1550 of `pandas/core/arrays/sparse/array.py`:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()
```

When `fill_value` is not null (e.g., 0 for integers):
1. `self.to_dense()` converts the sparse array to a dense numpy array
2. `SparseArray(...)` creates a NEW SparseArray from that dense array, which infers the same fill_value (0)
3. `.cumsum()` calls cumsum on the NEW SparseArray, which has the same `_null_fill_value=False`
4. This triggers the same code path again, causing infinite recursion

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         if axis is not None and axis >= self.ndim:  # Mimic ndarray behaviour.
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return SparseArray(np.cumsum(self.to_dense()))

         return SparseArray(
             self.sp_values.cumsum(),
```

The fix changes line 1550 to call numpy's `cumsum` directly on the dense array, then wrap the result in a SparseArray, rather than creating a SparseArray and calling its cumsum method recursively.