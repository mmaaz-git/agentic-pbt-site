# Bug Report: pandas.arrays.SparseArray.cumsum() Infinite Recursion

**Target**: `pandas.arrays.SparseArray.cumsum()`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `SparseArray.cumsum()` method causes infinite recursion when called on a SparseArray with a non-null fill value (e.g., fill_value=0 for integer arrays).

## Property-Based Test

```python
import pandas.arrays as pa
import numpy as np
from hypothesis import given, strategies as st, settings

def sparse_array_strategy(min_size=0, max_size=50):
    @st.composite
    def _strat(draw):
        size = draw(st.integers(min_value=min_size, max_value=max_size))
        fill_value = draw(st.sampled_from([0, 0.0, -1, 1]))
        values = draw(st.lists(
            st.sampled_from([fill_value, 1, 2, 3, -1, 10]),
            min_size=size, max_size=size
        ))
        return pa.SparseArray(values, fill_value=fill_value)
    return _strat()


@given(sparse_array_strategy(min_size=1, max_size=20))
@settings(max_examples=100)
def test_sparsearray_cumsum_doesnt_crash(arr):
    result = arr.cumsum()
    assert isinstance(result, pa.SparseArray)
```

**Failing input**: `SparseArray([0], fill_value=0)` (or any sparse array with non-null fill value)

## Reproducing the Bug

```python
import pandas.arrays as pa

sparse = pa.SparseArray([0], fill_value=0)
result = sparse.cumsum()
```

This triggers infinite recursion and crashes with `RecursionError: maximum recursion depth exceeded`.

## Why This Is A Bug

The `cumsum()` method is documented as "Cumulative sum of non-NA/null values" and should work on any SparseArray. The bug occurs because when `_null_fill_value` is False (i.e., the fill value is not NA/null, like 0 for integers), the code path is:

```python
return SparseArray(self.to_dense()).cumsum()
```

This creates a new SparseArray from the dense array and calls `cumsum()` on it recursively. Since the new SparseArray still has a non-null fill value, it enters the same code path again, causing infinite recursion.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         if axis is not None and axis >= self.ndim:  # Mimic ndarray behaviour.
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return SparseArray(self.to_dense().cumsum(), fill_value=np.nan)

         return SparseArray(
             self.sp_values.cumsum(),
```

The fix calls `cumsum()` on the dense numpy array instead of the SparseArray wrapper, and uses `np.nan` as the fill value as documented ("the fill value will be `np.nan` regardless").