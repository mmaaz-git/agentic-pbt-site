# Bug Report: pandas.core.sparse.SparseArray argmin/argmax Empty Values

**Target**: `pandas.core.arrays.sparse.SparseArray.argmin()` and `SparseArray.argmax()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When all values in a SparseArray are equal to the fill_value (resulting in zero sparse values), calling `argmin()` or `argmax()` raises a ValueError instead of returning a valid index.

## Property-Based Test

```python
@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    st.integers(min_value=-100, max_value=100)
)
@settings(max_examples=300)
def test_argmin_argmax_consistency(data, fill_value):
    arr = SparseArray(data, fill_value=fill_value)
    dense = arr.to_dense()

    if len(arr) > 0:
        assert arr[arr.argmin()] == dense[dense.argmin()]
        assert arr[arr.argmax()] == dense[dense.argmax()]
```

**Failing input**: `data=[0], fill_value=0`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.arrays.sparse import SparseArray

arr = SparseArray([0], fill_value=0)
dense = arr.to_dense()

print(f"SparseArray: {arr}")
print(f"Dense array: {dense}")
print(f"npoints: {arr.npoints}")

arr.argmin()
```

Output:
```
ValueError: attempt to get argmin of an empty sequence
```

The same error occurs with `argmax()`.

This also happens with any array where all values equal the fill_value:
```python
arr = SparseArray([5, 5, 5, 5], fill_value=5)
arr.argmin()
```

## Why This Is A Bug

When a SparseArray has all values equal to the fill_value, it is logically equivalent to a dense array of all the same values. In NumPy, calling `argmin()` or `argmax()` on such an array returns 0 (the first index), not an error:

```python
np.array([5, 5, 5, 5]).argmin()
```

The SparseArray implementation should maintain the same behavior as dense arrays for consistency. Users should not need to check if `npoints > 0` before calling these reduction operations.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1655,6 +1655,12 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]

+        # Handle case where all values are fill_value (no sparse values)
+        if len(non_nans) == 0:
+            _loc = self._first_fill_value_loc()
+            # If _loc is -1, there are no values at all, which shouldn't happen
+            return _loc if _loc >= 0 else 0
+
         _candidate = non_nan_idx[func(non_nans)]
         candidate = index[_candidate]