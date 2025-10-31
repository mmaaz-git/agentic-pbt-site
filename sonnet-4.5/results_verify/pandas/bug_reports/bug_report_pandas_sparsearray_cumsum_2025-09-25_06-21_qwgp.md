# Bug Report: SparseArray.cumsum() Infinite Recursion

**Target**: `pandas.core.arrays.sparse.array.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method on `SparseArray` causes infinite recursion when called on arrays with non-null fill values (e.g., `fill_value=0`), leading to `RecursionError`.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from pandas.core.arrays.sparse.array import SparseArray
import numpy as np
import pytest

@st.composite
def sparse_arrays(draw):
    size = draw(st.integers(min_value=0, max_value=100))
    values = draw(st.lists(
        st.integers(min_value=-1000, max_value=1000),
        min_size=size,
        max_size=size
    ))
    return SparseArray(values, dtype=np.int64)

@given(sparse_arrays())
def test_cumsum_consistency(sparse):
    try:
        sparse_cumsum = sparse.cumsum()
        dense_cumsum = sparse.to_dense().cumsum()
        result = sparse_cumsum.to_dense()
        np.testing.assert_allclose(result, dense_cumsum, rtol=1e-10)
    except (ValueError, TypeError) as e:
        pytest.skip(f"cumsum not supported for this input: {e}")
```

**Failing input**: `SparseArray([1])` (or any non-empty array with non-null fill value)

## Reproducing the Bug

```python
from pandas.core.arrays.sparse.array import SparseArray

sparse = SparseArray([1])
result = sparse.cumsum()
```

**Output**:
```
RecursionError: maximum recursion depth exceeded
```

## Why This Is A Bug

The bug is in line 1550 of `pandas/core/arrays/sparse/array.py`:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()
```

This creates a new `SparseArray` from the dense representation and then calls `cumsum()` on it, which triggers the same code path again, leading to infinite recursion. The method should call `cumsum()` on the **numpy array**, not on a new `SparseArray`.

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
