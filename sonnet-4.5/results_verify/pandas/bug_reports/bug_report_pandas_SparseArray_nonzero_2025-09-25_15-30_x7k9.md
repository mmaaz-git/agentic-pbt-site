# Bug Report: pandas.core.sparse.SparseArray.nonzero() Returns Incorrect Indices When fill_value Is Non-Zero

**Target**: `pandas.core.arrays.sparse.SparseArray.nonzero()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `nonzero()` method of `SparseArray` returns incorrect indices when the `fill_value` is non-zero. It only checks sparse values for non-zero, ignoring that the fill_value itself might be non-zero and should be included in the result.

## Property-Based Test

```python
import numpy as np
from pandas.core.arrays.sparse import SparseArray
from hypothesis import given, strategies as st


@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    st.integers(min_value=-100, max_value=100)
)
def test_nonzero_consistency(data, fill_value):
    sparse = SparseArray(data, fill_value=fill_value)
    dense = np.array(data)

    sparse_nonzero = sparse.nonzero()[0]
    dense_nonzero = dense.nonzero()[0]

    np.testing.assert_array_equal(sparse_nonzero, dense_nonzero)
```

**Failing input**: `data=[1], fill_value=1`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.arrays.sparse import SparseArray

data = [2, 2, 0, 2, 5]
fill_value = 2
sparse = SparseArray(data, fill_value=fill_value)
dense = np.array(data)

print(f"Sparse nonzero(): {sparse.nonzero()[0]}")
print(f"Dense nonzero():  {dense.nonzero()[0]}")

assert np.array_equal(sparse.nonzero()[0], dense.nonzero()[0])
```

Expected output: `[0, 1, 3, 4]` (all indices where value != 0)
Actual output: `[4]` (only the sparse index where sp_value != 0)

## Why This Is A Bug

The `nonzero()` method should return indices where the array values are non-zero, matching NumPy's behavior. Currently, when `fill_value` is non-zero (e.g., 2), the method only checks if the explicitly stored sparse values are non-zero, completely ignoring that positions containing the fill_value are also non-zero.

In the example `[2, 2, 0, 2, 5]` with `fill_value=2`:
- Sparse representation stores only `[0, 5]` at indices `[2, 4]`
- Positions 0, 1, 3 implicitly contain the value 2 (fill_value)
- All positions except index 2 contain non-zero values
- Expected `nonzero()`: `[0, 1, 3, 4]`
- Actual `nonzero()`: `[4]`

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1410,7 +1410,13 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
     def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
         if self.fill_value == 0:
             return (self.sp_index.indices,)
         else:
-            return (self.sp_index.indices[self.sp_values != 0],)
+            # When fill_value is non-zero, we need to account for:
+            # 1. Sparse positions where sp_values != 0
+            # 2. Fill positions (which are non-zero by definition)
+            mask = np.ones(len(self), dtype=bool)
+            mask[self.sp_index.indices] = self.sp_values != 0
+            return (np.flatnonzero(mask).astype(np.int32),)