# Bug Report: pandas.core.arrays.sparse.SparseArray.sum() ignores NaN in sp_values when skipna=False

**Target**: `pandas.core.arrays.sparse.SparseArray.sum()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `SparseArray.sum(skipna=False)` is called and the sparse values contain NaN, it incorrectly returns a numeric result instead of NaN. This violates the expected pandas behavior where `skipna=False` should propagate NaN values in reduction operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.arrays.sparse import SparseArray

@given(st.lists(
    st.floats(allow_nan=True, allow_infinity=False) |
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=100
))
def test_sum_matches_dense_skipna_false(data):
    arr = SparseArray(data, fill_value=0.0)

    sparse_sum = arr.sum(skipna=False)
    dense_sum = arr.to_dense().sum()

    if np.isnan(dense_sum):
        assert np.isnan(sparse_sum), f"Expected NaN but got {sparse_sum}"
    else:
        assert sparse_sum == dense_sum
```

**Failing input**: `[np.nan]` with `fill_value=0.0`

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import SparseArray

arr = SparseArray([np.nan], fill_value=0.0)

print("Expected (pandas Series):")
s = pd.Series([np.nan])
print(f"  pd.Series([np.nan]).sum(skipna=False) = {s.sum(skipna=False)}")

print("\nActual (SparseArray):")
print(f"  SparseArray([np.nan]).sum(skipna=False) = {arr.sum(skipna=False)}")
```

**Output:**
```
Expected (pandas Series):
  pd.Series([np.nan]).sum(skipna=False) = nan

Actual (SparseArray):
  SparseArray([np.nan]).sum(skipna=False) = 0.0
```

## Why This Is A Bug

The `sum()` method should behave consistently with pandas Series. When `skipna=False`, any NaN values in the data should cause the sum to return NaN. Currently, the method only detects NaN values in "gaps" (positions with fill_value), but fails to detect NaN values that are explicitly stored in `sp_values`.

The issue is in the logic at line 1511-1513 of `array.py`:

```python
has_na = self.sp_index.ngaps > 0 and not self._null_fill_value

if has_na and not skipna:
    return na_value_for_dtype(self.dtype.subtype, compat=False)
```

This only checks if there are gaps with a non-null fill value, but doesn't check if `sp_values` itself contains NaN values.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1506,10 +1506,16 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         scalar
         """
         nv.validate_sum(args, kwargs)
         valid_vals = self._valid_sp_values
         sp_sum = valid_vals.sum()
         has_na = self.sp_index.ngaps > 0 and not self._null_fill_value
+
+        # Also check if sp_values contains any NaN values
+        if not skipna and len(valid_vals) < len(self.sp_values):
+            # Some sp_values were filtered out as NaN
+            return na_value_for_dtype(self.dtype.subtype, compat=False)

         if has_na and not skipna:
             return na_value_for_dtype(self.dtype.subtype, compat=False)