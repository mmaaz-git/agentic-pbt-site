# Bug Report: pandas.core.sparse.SparseArray.mean Incorrect with NaN Fill Value

**Target**: `pandas.core.sparse.SparseArray.mean`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `mean()` method of `SparseArray` returns an incorrect result when the fill value is NaN and the sparse values array contains NaN values. It ignores NaN values in `sp_values` when computing the mean, returning a non-NaN result when it should return NaN.

## Property-Based Test

```python
from pandas.arrays import SparseArray
import numpy as np
from hypothesis import given, strategies as st, assume
import math

@st.composite
def sparse_arrays(draw):
    size = draw(st.integers(min_value=1, max_value=100))
    fill_value = np.nan
    elements = st.floats(allow_nan=True, allow_infinity=False)
    data = draw(st.lists(elements, min_size=size, max_size=size))
    return SparseArray(data, fill_value=fill_value)

@given(sparse_arrays())
def test_mean_matches_dense(arr):
    assume(len(arr) > 0)
    sparse_mean = arr.mean()
    dense_mean = arr.to_dense().mean()

    if isinstance(sparse_mean, float) and np.isnan(sparse_mean):
        assert np.isnan(dense_mean)
    else:
        assert math.isclose(sparse_mean, dense_mean, rel_tol=1e-9, abs_tol=1e-9)
```

**Failing input**: `SparseArray([0.0, np.nan], fill_value=np.nan)`

## Reproducing the Bug

```python
from pandas.arrays import SparseArray
import numpy as np

arr = SparseArray([0.0, np.nan], fill_value=np.nan)
sparse_mean = arr.mean()
dense_mean = arr.to_dense().mean()

print(f"Sparse mean: {sparse_mean}")
print(f"Dense mean: {dense_mean}")
```

Output:
```
Sparse mean: 0.0
Dense mean: nan
```

## Why This Is A Bug

When computing the mean of an array containing NaN values, the standard NumPy behavior is to return NaN. The SparseArray should match this behavior for consistency. Currently, when `fill_value` is NaN, the implementation only considers non-NaN values in `sp_values`, but ignores NaN values that are explicitly stored in `sp_values`, leading to incorrect results.

## Fix

The issue is in the `mean` method (lines 1558-1575). When `_null_fill_value` is True, it should check if any of the `sp_values` are NaN:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1566,10 +1566,14 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         nv.validate_mean(args, kwargs)
         valid_vals = self._valid_sp_values
         sp_sum = valid_vals.sum()
         ct = len(valid_vals)

         if self._null_fill_value:
+            # If any sp_values are NaN, the mean should be NaN
+            if len(self.sp_values) > len(valid_vals):
+                return na_value_for_dtype(self.dtype.subtype, compat=False)
             return sp_sum / ct
         else:
             nsparse = self.sp_index.ngaps
             return (sp_sum + self.fill_value * nsparse) / (ct + nsparse)
```