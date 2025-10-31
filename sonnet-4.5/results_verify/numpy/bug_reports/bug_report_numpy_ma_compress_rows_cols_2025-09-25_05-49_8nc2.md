# Bug Report: numpy.ma.compress_rows/compress_cols Dimensionality Loss

**Target**: `numpy.ma.compress_rows` and `numpy.ma.compress_cols`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When all rows/columns are masked, `compress_rows` and `compress_cols` return a 1D empty array instead of a 2D array with one dimension being 0, breaking dimensionality invariants and causing inconsistency with numpy's own `compress` function.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as npst


@given(
    npst.arrays(
        dtype=np.float64,
        shape=(5, 5),
        elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
    ),
    npst.arrays(dtype=np.bool_, shape=(5, 5))
)
def test_compress_rows_preserves_dimensionality(arr, mask):
    masked_arr = ma.array(arr, mask=mask)
    result = ma.compress_rows(masked_arr)
    assert result.ndim == 2


@given(
    npst.arrays(
        dtype=np.float64,
        shape=(5, 5),
        elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
    ),
    npst.arrays(dtype=np.bool_, shape=(5, 5))
)
def test_compress_cols_preserves_dimensionality(arr, mask):
    masked_arr = ma.array(arr, mask=mask)
    result = ma.compress_cols(masked_arr)
    assert result.ndim == 2
```

**Failing input**: 5×5 array with all values masked

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

arr = np.array([[1., 2., 3.],
                [4., 5., 6.]])
mask_all = np.ones((2, 3), dtype=bool)
masked_arr = ma.array(arr, mask=mask_all)

result_rows = ma.compress_rows(masked_arr)
print(f"compress_rows shape: {result_rows.shape}, ndim: {result_rows.ndim}")

result_cols = ma.compress_cols(masked_arr)
print(f"compress_cols shape: {result_cols.shape}, ndim: {result_cols.ndim}")

np_result = np.compress([False, False], arr, axis=0)
print(f"numpy.compress shape: {np_result.shape}, ndim: {np_result.ndim}")
```

Output:
```
compress_rows shape: (0,), ndim: 1
compress_cols shape: (0,), ndim: 1
numpy.compress shape: (0, 3), ndim: 2
```

## Why This Is A Bug

The functions `compress_rows` and `compress_cols` are documented to work on 2D arrays and suppress rows/columns containing masked values. When all rows/columns are suppressed, the expected behavior is to return an empty 2D array with the appropriate shape (e.g., `(0, 3)` for compress_rows or `(3, 0)` for compress_cols), not a 1D array.

This violates:
1. **Dimensionality preservation**: Operations on 2D arrays should return 2D arrays
2. **Consistency**: numpy's `compress` preserves dimensionality, but `ma.compress_rows/cols` does not
3. **Type stability**: Code expecting 2D arrays will fail with `IndexError` when trying to access `result.shape[1]`

## Fix

The issue originates in `compress_rowcols` in `/numpy/ma/extras.py`. When the result is empty, it should preserve the 2D structure:

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -945,7 +945,12 @@ def compress_rowcols(x, axis=None):
         if axis in [1, -1]:
             return compress_cols(x)
         return x[idxr][:, idxc]
-    return x[[]]
+    # Preserve 2D shape when result is empty
+    if axis == 0:  # rows
+        return x[[]].reshape(0, x.shape[1])
+    elif axis in [1, -1]:  # cols
+        return x[:, []].reshape(x.shape[0], 0)
+    return x[[]][:, []]
```