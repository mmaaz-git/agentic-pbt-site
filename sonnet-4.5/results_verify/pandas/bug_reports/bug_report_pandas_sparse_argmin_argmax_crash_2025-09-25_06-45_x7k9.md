# Bug Report: SparseArray argmin/argmax Crash on All-Fill-Value Arrays

**Target**: `pandas.core.arrays.sparse.SparseArray.argmin` and `pandas.core.arrays.sparse.SparseArray.argmax`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`SparseArray.argmin()` and `SparseArray.argmax()` crash with `ValueError: attempt to get argmin of an empty sequence` when all array values equal the fill value, instead of returning the index of the first occurrence like numpy does.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.arrays.sparse import SparseArray

@given(
    st.lists(
        st.integers(min_value=-100, max_value=100),
        min_size=2,
        max_size=50,
    )
)
def test_argmin_argmax_consistency(values):
    arr = np.array(values)
    sparse = SparseArray(arr)

    assert sparse.argmin() == np.argmin(arr)
    assert sparse.argmax() == np.argmax(arr)
```

**Failing input**: `[0, 0]` (all values equal the default fill_value of 0)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.arrays.sparse import SparseArray

arr = np.array([0, 0])
sparse = SparseArray(arr)

print(f"Array: {arr}")
print(f"Fill value: {sparse.fill_value}")
print(f"Sparse values: {sparse.sp_values}")

print(f"np.argmin(arr): {np.argmin(arr)}")

sparse.argmin()
```

**Output:**
```
Array: [0 0]
Fill value: 0
Sparse values: []
np.argmin(arr): 0
ValueError: attempt to get argmin of an empty sequence
```

## Why This Is A Bug

When all values in a SparseArray equal the fill value, `sp_values` is empty (no non-fill values are stored). The `_argmin_argmax` method tries to compute argmin/argmax on this empty array at line 1658, causing a crash:

```python
_candidate = non_nan_idx[func(non_nans)]  # func is np.argmin or np.argmax
```

When `non_nans` is empty, numpy raises `ValueError`. The method should instead recognize this case and return the index of the first fill value (which would be returned by `_first_fill_value_loc()`).

The bug violates the API contract that `SparseArray` methods should behave like their dense counterparts. `np.argmin([0, 0])` correctly returns `0`, but `SparseArray([0, 0]).argmin()` crashes.

## Fix

```diff
diff --git a/pandas/core/arrays/sparse/array.py b/pandas/core/arrays/sparse/array.py
index 1234567..abcdefg 100644
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1653,6 +1653,13 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         mask = np.asarray(isna(values))
         func = np.argmax if kind == "argmax" else np.argmin

+        # Handle case where all values are fill_value (sp_values may be empty or all NaN)
+        non_nans = values[~mask]
+        if len(non_nans) == 0:
+            # All stored values are NaN, return first fill value location
+            return self._first_fill_value_loc() if self._first_fill_value_loc() != -1 else 0
+
         idx = np.arange(values.shape[0])
-        non_nans = values[~mask]
         non_nan_idx = idx[~mask]

         _candidate = non_nan_idx[func(non_nans)]
```