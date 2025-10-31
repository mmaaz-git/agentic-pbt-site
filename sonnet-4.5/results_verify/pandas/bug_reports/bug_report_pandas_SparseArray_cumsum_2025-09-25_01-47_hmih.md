# Bug Report: pandas.arrays.SparseArray.cumsum Infinite Recursion

**Target**: `pandas.arrays.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

SparseArray.cumsum() causes infinite recursion and crashes when the fill_value is not NaN/null.

## Property-Based Test

```python
import pandas.arrays as pa
from hypothesis import given, strategies as st

@st.composite
def sparse_arrays(draw):
    size = draw(st.integers(min_value=0, max_value=50))
    fill_value = draw(st.integers(min_value=-10, max_value=10))
    data = draw(st.lists(st.integers(min_value=-100, max_value=100), min_size=size, max_size=size))
    return pa.SparseArray(data, fill_value=fill_value)

@given(sparse_arrays())
def test_sparse_array_cumsum_length(arr):
    result = arr.cumsum()
    assert len(result) == len(arr)
```

**Failing input**: `SparseArray([1, 0, 2, 0, 3], fill_value=0)`

## Reproducing the Bug

```python
import pandas.arrays as pa

arr = pa.SparseArray([1, 0, 2, 0, 3], fill_value=0)
result = arr.cumsum()
```

## Why This Is A Bug

The cumsum() method has a code path that creates infinite recursion:

```python
def cumsum(self, ...):
    if not self._null_fill_value:
        return SparseArray(self.to_dense()).cumsum()
```

When `_null_fill_value` is False (fill_value is not NaN), it creates a new SparseArray from the dense representation and calls cumsum() on it. The new SparseArray has the same fill_value, so it takes the same code path, leading to infinite recursion until RecursionError is raised.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,8 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         if axis is not None and axis >= self.ndim:
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            dense_result = np.asarray(self).cumsum()
+            return SparseArray(dense_result, fill_value=np.nan)

         return SparseArray(
             self.sp_values.cumsum(),
```