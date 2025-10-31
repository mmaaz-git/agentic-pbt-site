# Bug Report: numpy.ma compress_rows/compress_cols/compress_rowcols Dimensionality Loss

**Target**: `numpy.ma.compress_rows`, `numpy.ma.compress_cols`, `numpy.ma.compress_rowcols`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The compress_rows, compress_cols, and compress_rowcols functions lose dimensionality when all rows/columns are removed, returning 1D arrays instead of maintaining 2D structure.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

@st.composite
def masked_2d_arrays(draw):
    rows = draw(st.integers(min_value=1, max_value=5))
    cols = draw(st.integers(min_value=1, max_value=5))
    data = draw(npst.arrays(dtype=np.float64, shape=(rows, cols),
                           elements=st.floats(allow_nan=False, allow_infinity=False,
                                            min_value=-100, max_value=100)))
    mask = draw(npst.arrays(dtype=bool, shape=(rows, cols)))
    return ma.array(data, mask=mask)

@given(masked_2d_arrays())
@settings(max_examples=500)
def test_compress_rows_preserves_2d_structure(arr):
    result = ma.compress_rows(arr)
    assert result.ndim == 2
```

**Failing input**: `masked_array(data=[[--]], mask=[[True]])`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

arr = ma.array([[999]], mask=[[True]])

result_rows = ma.compress_rows(arr)
result_cols = ma.compress_cols(arr)
result_rowcols = ma.compress_rowcols(arr)

print(f"Input shape: {arr.shape} (2D)")
print(f"compress_rows result: {result_rows.shape}, ndim={result_rows.ndim}")
print(f"compress_cols result: {result_cols.shape}, ndim={result_cols.ndim}")
print(f"compress_rowcols result: {result_rowcols.shape}, ndim={result_rowcols.ndim}")

arr2 = ma.array([[1, 2], [3, 4]], mask=[[True, True], [True, True]])
result2 = ma.compress_rows(arr2)
print(f"Multi-column input: {arr2.shape}")
print(f"compress_rows result: {result2.shape}, ndim={result2.ndim}")
```

## Why This Is A Bug

The documentation states these functions:
- Take a 2D array as input ("Must be a 2D array")
- Return "the compressed array"

When the result is non-empty, all three functions correctly return 2D arrays. However, when all rows/columns are removed, they inconsistently return 1D arrays with shape `(0,)` instead of maintaining 2D structure with shapes like `(0, n)` or `(n, 0)`.

This inconsistency breaks downstream code that expects 2D output:
```python
result = ma.compress_rows(arr)
num_cols = result.shape[1]
```

This code works fine normally but crashes with `IndexError: tuple index out of range` when all rows are masked in a single-column array.

## Fix

The issue appears to be in the implementation (likely in extras.py:948). The functions should preserve 2D structure for empty results:

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -945,7 +945,7 @@ def compress_rows(x):
     result_rows = np.flatnonzero(~(mask.sum(axis=1).astype(bool)))
     if not result_rows.size:
-        return result_rows
+        return np.empty((0, x.shape[1]), dtype=x.dtype)
     return x[result_rows]
```

Similar fixes needed for compress_cols (return shape `(x.shape[0], 0)`) and compress_rowcols (return shape `(0, 0)`) when appropriate.