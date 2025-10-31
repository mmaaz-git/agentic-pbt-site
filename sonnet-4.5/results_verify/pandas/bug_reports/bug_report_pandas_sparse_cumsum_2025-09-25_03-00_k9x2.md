# Bug Report: pandas.core.arrays.sparse.SparseArray.cumsum Infinite Recursion

**Target**: `pandas.core.arrays.sparse.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method on `SparseArray` causes infinite recursion when called on arrays with non-null fill values (e.g., `fill_value=0`).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.core.arrays.sparse as sparse
import numpy as np

@given(
    data=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    fill_value=st.integers(min_value=-10, max_value=10).filter(lambda x: x not in [np.nan])
)
def test_sparse_array_cumsum_should_not_crash(data, fill_value):
    """cumsum should work on sparse arrays with non-null fill values"""
    np_data = np.array(data)
    sparse_arr = sparse.SparseArray(np_data, fill_value=fill_value)

    result = sparse_arr.cumsum()
    assert len(result) == len(sparse_arr)
```

**Failing input**: Any `SparseArray` with a non-null fill value, e.g., `SparseArray([0, 1, 2], fill_value=0)`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

data = np.array([0, 1, 2])
sparse_arr = pd.arrays.SparseArray(data, fill_value=0)

result = sparse_arr.cumsum()
```

This raises `RecursionError: maximum recursion depth exceeded`.

Expected result: `SparseArray([0, 1, 3], fill_value=0)`

## Why This Is A Bug

The `cumsum()` method has a code path for non-null fill values that creates infinite recursion:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()
```

This line converts to dense, wraps it in a new `SparseArray`, then calls `cumsum()` on that new array. However, the new `SparseArray` created from `self.to_dense()` inherits the same non-null fill value (since `SparseArray.__init__` infers it from the data), causing `_null_fill_value` to be False again. This triggers the same code path, leading to infinite recursion.

The cumsum operation is a fundamental array operation that users expect to work. The docstring promises it will return a `SparseArray` with cumulative sum, but it crashes instead.

## Fix

The fix is to call `cumsum()` on the dense array **before** wrapping it in a `SparseArray`:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         if axis is not None and axis >= self.ndim:  # Mimic ndarray behaviour.
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return SparseArray(self.to_dense().cumsum())

         return SparseArray(
             self.sp_values.cumsum(),
```

This computes the cumsum on the dense numpy array (which works correctly), then wraps the result in a `SparseArray`.