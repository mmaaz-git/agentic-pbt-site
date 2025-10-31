# Bug Report: numpy.ma.compress_nd Family Dimensionality Inconsistency

**Target**: `numpy.ma.compress_nd`, `numpy.ma.compress_rowcols`, `numpy.ma.compress_rows`, `numpy.ma.compress_cols`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `compress_nd` function family returns inconsistent array dimensions: fully masked arrays return 1D empty arrays, while partially masked arrays that result in complete removal return properly shaped empty arrays.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

@st.composite
def masked_2d_arrays(draw):
    shape = draw(npst.array_shapes(min_dims=2, max_dims=2, max_side=10))
    data = draw(npst.arrays(dtype=np.int64, shape=shape,
                           elements=st.integers(min_value=-100, max_value=100)))
    mask = draw(npst.arrays(dtype=bool, shape=shape))
    return ma.array(data, mask=mask)

@given(masked_2d_arrays())
@settings(max_examples=1000)
def test_compress_rowcols_maintains_2d(arr):
    result = ma.compress_rowcols(arr)
    assert result.ndim == 2
```

**Failing input**: `masked_array(data=[[--]], mask=[[True]])`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

arr1 = ma.array([[99]], mask=[[True]])
result1 = ma.compress_rowcols(arr1)
print(f"compress_rowcols: shape={arr1.shape} -> {result1.shape}, ndim={arr1.ndim} -> {result1.ndim}")

result2 = ma.compress_rows(arr1)
print(f"compress_rows: shape={arr1.shape} -> {result2.shape}, ndim={arr1.ndim} -> {result2.ndim}")

result3 = ma.compress_cols(arr1)
print(f"compress_cols: shape={arr1.shape} -> {result3.shape}, ndim={arr1.ndim} -> {result3.ndim}")

arr2 = ma.array([[1, 2], [3, 4]], mask=[[True, False], [False, True]])
result4 = ma.compress_rowcols(arr2)
print(f"\nPartially masked (all removed): shape={arr2.shape} -> {result4.shape}, ndim={arr2.ndim} -> {result4.ndim}")
```

## Why This Is A Bug

The function exhibits inconsistent dimensionality behavior:
- Fully masked 2D array `[[--]]` → returns 1D array with shape `(0,)`
- Partially masked 2D array where all rows/columns get removed → returns 2D array with shape `(0, 0)`

The documentation states the input "Must be a 2D array" for `compress_rowcols`, and users expect dimension-preserving behavior. The output dimensionality should be consistent regardless of whether the array was fully masked initially or became empty through filtering.

## Fix

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -945,7 +945,12 @@ def compress_nd(x, axis=None):
     if m is nomask or not m.any():
         return x._data
     # All is masked: return empty
     if m.all():
-        return nxarray([])
+        # Preserve dimensionality - return empty array with appropriate shape
+        result_shape = list(x.shape)
+        for ax in axis:
+            result_shape[ax] = 0
+        return np.empty(result_shape, dtype=x.dtype)
     # Filter elements through boolean indexing
     data = x._data
     for ax in axis:
```