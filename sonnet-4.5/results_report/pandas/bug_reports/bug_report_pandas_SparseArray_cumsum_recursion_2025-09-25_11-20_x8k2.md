# Bug Report: pandas.SparseArray.cumsum Infinite Recursion

**Target**: `pandas.core.arrays.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Calling `cumsum()` on a SparseArray with a non-null fill value (e.g., fill_value=0) causes infinite recursion and crashes with RecursionError.

## Property-Based Test

```python
import pandas.core.arrays as arrays
from hypothesis import given, settings, strategies as st, assume

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=30))
@settings(max_examples=200)
def test_sparse_cumsum_monotonic_nonnegative(data):
    assume(all(x >= 0 for x in data))
    sparse = arrays.SparseArray(data)
    result = sparse.cumsum()
    dense_result = result.to_dense()

    for i in range(len(dense_result) - 1):
        assert dense_result[i] <= dense_result[i + 1]
```

**Failing input**: `data = [1, 2, 3]` (any non-empty list triggers it)

## Reproducing the Bug

```python
import pandas as pd

sparse = pd.arrays.SparseArray([1, 0, 2, 0, 3])
result = sparse.cumsum()
```

**Output:**
```
RecursionError: maximum recursion depth exceeded
```

## Why This Is A Bug

The implementation at `pandas/core/arrays/sparse/array.py:1550` contains infinite recursion:

```python
def cumsum(self, axis: AxisInt = 0, *args, **kwargs) -> SparseArray:
    ...
    if not self._null_fill_value:
        return SparseArray(self.to_dense()).cumsum()  # Bug: recursion!
```

When `_null_fill_value` is False (e.g., when fill_value=0), this:
1. Converts to dense: `self.to_dense()`
2. Creates new SparseArray: `SparseArray(...)`
3. Calls cumsum on it: `.cumsum()`
4. The new SparseArray ALSO has `_null_fill_value=False`, so it repeats steps 1-3 infinitely

This affects any SparseArray with a non-null fill_value, making `cumsum()` completely unusable for common cases like fill_value=0.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, ExtensionArray):
         if axis is not None and axis >= self.ndim:  # Mimic ndarray behaviour.
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return type(self)(self.to_dense().cumsum())

         return SparseArray(
             self.sp_values.cumsum(),
```

The fix converts to dense, performs cumsum on the dense array, then wraps the result in a SparseArray, avoiding the recursion.