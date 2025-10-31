# Bug Report: SparseArray max/min Incorrect Behavior with skipna=False

**Target**: `pandas.core.arrays.sparse.SparseArray._min_max`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `SparseArray.max()` or `SparseArray.min()` is called with `skipna=False` on an array containing NaN values and a non-null fill_value, the methods incorrectly ignore the NaN values and return a numeric result instead of NaN.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.arrays import SparseArray

@given(st.lists(
    st.one_of(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        st.just(np.nan)
    ),
    min_size=1,
    max_size=50
))
def test_max_min_skipna_false(values):
    arr = SparseArray(values, fill_value=0.0)

    if any(np.isnan(v) for v in values):
        sparse_max = arr.max(skipna=False)
        dense_max = np.max(arr.to_dense())

        assert np.isnan(sparse_max) and np.isnan(dense_max), \
            f"When array contains NaN and skipna=False, max should return NaN"
```

**Failing input**: `[1.0, nan, 0.0]` with `fill_value=0.0`

## Reproducing the Bug

```python
import numpy as np
from pandas.arrays import SparseArray

arr = SparseArray([1.0, np.nan, 0.0], fill_value=0.0)

result = arr.max(skipna=False)

print(f"Result: {result}")
print(f"Expected: nan")
```

Output:
```
Result: 1.0
Expected: nan
```

## Why This Is A Bug

According to pandas semantics and NumPy compatibility, when `skipna=False` and the array contains NaN values, aggregation functions like `max()` and `min()` should return NaN to indicate the presence of missing data. This is consistent with:

1. NumPy's behavior: `np.max([1.0, np.nan, 0.0])` returns `nan`
2. Dense pandas arrays: `pd.Series([1.0, np.nan, 0.0]).max(skipna=False)` returns `nan`
3. The documented contract: "skipna : bool, default True - Whether to ignore NA values."

The bug occurs because the `_min_max` method uses `self._valid_sp_values` which always filters out NaN values, regardless of the `skipna` parameter. When `has_nonnull_fill_vals` is True (meaning the array has a non-null fill value like 0.0), the method returns the max/min without checking if NaN values are present in the original `sp_values`.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1623,7 +1623,14 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         -------
         scalar
         """
-        valid_vals = self._valid_sp_values
+        # When skipna=True, filter out NaN values. When skipna=False, keep them.
+        if skipna:
+            valid_vals = self._valid_sp_values
+        else:
+            # Check if any NaN values exist in sp_values
+            if len(self.sp_values) > 0 and isna(self.sp_values).any():
+                return na_value_for_dtype(self.dtype.subtype, compat=False)
+            valid_vals = self.sp_values
         has_nonnull_fill_vals = not self._null_fill_value and self.sp_index.ngaps > 0

         if len(valid_vals) > 0: