# Bug Report: pandas.core.arrays.sparse - argmin/argmax crash on all-fill-value arrays

**Target**: `pandas.core.arrays.sparse.SparseArray.argmin()` and `pandas.core.arrays.sparse.SparseArray.argmax()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

SparseArray's `argmin()` and `argmax()` methods crash with `ValueError: attempt to get argmin of an empty sequence` when the array contains only fill values (i.e., when all values equal the fill_value).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.arrays import SparseArray

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    st.integers(min_value=-100, max_value=100),
)
def test_sparse_array_argmin_argmax_match_dense(data, fill_value):
    """
    Property: argmin() and argmax() should match dense array
    Evidence: _argmin_argmax method should find correct positions
    """
    sparse = SparseArray(data, fill_value=fill_value)
    dense = np.array(data)

    assert sparse.argmin() == dense.argmin()
    assert sparse.argmax() == dense.argmax()
```

**Failing input**: `data=[0], fill_value=0`

## Reproducing the Bug

```python
import numpy as np
from pandas.arrays import SparseArray

data = [0]
fill_value = 0

sparse = SparseArray(data, fill_value=fill_value)
dense = np.array(data)

print(dense.argmin())  # 0

sparse.argmin()  # ValueError: attempt to get argmin of an empty sequence
```

## Why This Is A Bug

When a SparseArray contains only fill values, the internal `sp_values` array is empty. The `_argmin_argmax` method at line 1658 in `array.py` attempts to call `np.argmin(non_nans)` or `np.argmax(non_nans)` on an empty array, which raises a ValueError.

The method should instead detect this case early and return the location of the first fill value (which would be 0 for an all-fill-value array), matching NumPy's behavior.

This violates the documented behavior that these methods should behave like their NumPy counterparts.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1648,11 +1648,18 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
     def _argmin_argmax(self, kind: Literal["argmin", "argmax"]) -> int:
         values = self._sparse_values
         index = self._sparse_index.indices
         mask = np.asarray(isna(values))
         func = np.argmax if kind == "argmax" else np.argmin

         idx = np.arange(values.shape[0])
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]

+        # Handle case where all values are fill_value (no sparse values)
+        if len(non_nans) == 0:
+            _loc = self._first_fill_value_loc()
+            return _loc if _loc != -1 else 0
+
         _candidate = non_nan_idx[func(non_nans)]
         candidate = index[_candidate]

         if isna(self.fill_value):
             return candidate
         if kind == "argmin" and self[candidate] < self.fill_value:
             return candidate
         if kind == "argmax" and self[candidate] > self.fill_value:
             return candidate
         _loc = self._first_fill_value_loc()
         if _loc == -1:
             # fill_value doesn't exist
             return candidate
         else:
             return _loc
```