# Bug Report: pandas.core.arrays.sparse.SparseArray.argmax/argmin crash on all-fill arrays

**Target**: `pandas.core.arrays.sparse.SparseArray.argmax()` and `pandas.core.arrays.sparse.SparseArray.argmin()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When calling `argmax()` or `argmin()` on a SparseArray where all values equal the fill_value, the methods crash with `ValueError: attempt to get argmax of an empty sequence`. This happens because the implementation tries to find the argmax/argmin of an empty `sp_values` array.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.arrays.sparse import SparseArray

@given(
    data=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    fill_value=st.integers(min_value=-1000, max_value=1000)
)
def test_sparse_array_argmax_argmin_match_dense(data, fill_value):
    arr = np.array(data)
    sparse = SparseArray(arr, fill_value=fill_value)

    assert sparse.argmax() == arr.argmax()
    assert sparse.argmin() == arr.argmin()
```

**Failing input**: `data=[0], fill_value=0`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.arrays.sparse import SparseArray

data = [0]
sparse = SparseArray(data, fill_value=0)

print(sparse.argmax())
```

Output:
```
ValueError: attempt to get argmax of an empty sequence
```

## Why This Is A Bug

The `argmax()` and `argmin()` methods should return the index of the maximum/minimum value in the array, just like NumPy's argmax/argmin. When all values are equal (specifically, all equal to the fill_value), the method should return 0 (the first index), not crash.

The crash occurs because:
1. When all array values equal the fill_value, `sp_values` is empty (no non-fill values to store)
2. The `_argmin_argmax` method at line 1658 tries to compute `non_nan_idx[func(non_nans)]`
3. When `non_nans` is empty, calling `np.argmax()` or `np.argmin()` on it raises a ValueError

This is inconsistent with NumPy's behavior, where `np.array([0]).argmax()` returns `0`.

## Fix

The issue is in the `_argmin_argmax` method at lines 1648-1672 of `/pandas/core/arrays/sparse/array.py`. The method needs to handle the case where all values are fill values.

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1652,6 +1652,11 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         func = np.argmax if kind == "argmax" else np.argmin

         idx = np.arange(values.shape[0])
+
+        # If all values are NA or all values are fill_value, return fill_value location
+        if len(values) == 0 or mask.all():
+            return self._first_fill_value_loc() if self._first_fill_value_loc() != -1 else 0
+
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]