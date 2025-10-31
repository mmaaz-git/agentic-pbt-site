# Bug Report: pandas.core.sparse.SparseArray argmin/argmax Crash on Fill-Only Arrays

**Target**: `pandas.core.sparse.SparseArray.argmin` and `pandas.core.sparse.SparseArray.argmax`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `argmin()` and `argmax()` methods crash with `ValueError: attempt to get argmin of an empty sequence` when called on SparseArrays where all values equal the fill value.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.arrays import SparseArray
import numpy as np

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50))
def test_argmin_argmax_consistent_with_dense(data):
    sparse = SparseArray(data)
    dense = sparse.to_dense()

    assert sparse.argmin() == np.argmin(dense)
    assert sparse.argmax() == np.argmax(dense)
```

**Failing input**: `[0]` (any list containing only the default fill value)

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

sparse = SparseArray([0])
print(sparse.argmin())
```

Output:
```
ValueError: attempt to get argmin of an empty sequence
```

## Why This Is A Bug

When all values in a SparseArray equal the fill value (e.g., `[0]` with default fill value 0), the sparse representation stores zero non-fill values. The `_argmin_argmax` method tries to compute `np.argmin(non_nans)` where `non_nans` is an empty array, causing the crash.

This is inconsistent with numpy's behavior: `np.argmin([0])` correctly returns `0`.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1655,7 +1655,12 @@ class SparseArray(OpsMixin, ExtensionArray):
         idx = np.arange(values.shape[0])
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]
-
-        _candidate = non_nan_idx[func(non_nans)]
+
+        if len(non_nans) == 0:
+            # All values are fill_value, return first location
+            _loc = self._first_fill_value_loc()
+            return _loc if _loc != -1 else 0
+
+        _candidate = non_nan_idx[func(non_nans)]
         candidate = index[_candidate]

         if isna(self.fill_value):
```

The fix checks if all stored values are NaN/masked, and if so, returns the first fill value location directly.