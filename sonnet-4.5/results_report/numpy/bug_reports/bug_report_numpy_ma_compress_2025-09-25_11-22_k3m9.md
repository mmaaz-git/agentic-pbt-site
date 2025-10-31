# Bug Report: numpy.ma.compress_rows and compress_cols Shape Inconsistency

**Target**: `numpy.ma.compress_rows`, `numpy.ma.compress_cols`, `numpy.ma.compress_rowcols`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`compress_rows` and `compress_cols` return inconsistent shapes when operating on fully-masked 2-D arrays. When all values in the input array are masked, these functions return a 1-D array with shape `(0,)` instead of maintaining the 2-D structure with shapes `(0, cols)` or `(rows, 0)` respectively.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st

@given(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10))
def test_compress_rows_shape_inconsistency_when_all_masked(rows, cols):
    data = np.zeros((rows, cols))
    mask = np.ones((rows, cols), dtype=bool)

    arr = ma.array(data, mask=mask)
    result = ma.compress_rows(arr)

    assert result.ndim == 2
    assert result.shape == (0, cols)

@given(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10))
def test_compress_cols_shape_inconsistency_when_all_masked(rows, cols):
    data = np.zeros((rows, cols))
    mask = np.ones((rows, cols), dtype=bool)

    arr = ma.array(data, mask=mask)
    result = ma.compress_cols(arr)

    assert result.ndim == 2
    assert result.shape == (rows, 0)
```

**Failing input**: Any 2-D array where all elements are masked

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

data = np.array([[1., 2., 3.],
                 [4., 5., 6.]])
mask = np.ones((2, 3), dtype=bool)
arr = ma.array(data, mask=mask)

result_rows = ma.compress_rows(arr)
print(f"compress_rows shape: {result_rows.shape}")
print(f"Expected: (0, 3), Actual: {result_rows.shape}")

result_cols = ma.compress_cols(arr)
print(f"compress_cols shape: {result_cols.shape}")
print(f"Expected: (2, 0), Actual: {result_cols.shape}")

data_partial = np.array([[1., 2., 3.],
                        [4., 5., 6.]])
mask_partial = np.array([[True, False, False],
                        [False, True, False]])
arr_partial = ma.array(data_partial, mask=mask_partial)

result_rows_partial = ma.compress_rows(arr_partial)
print(f"compress_rows (partial mask) shape: {result_rows_partial.shape}")

result_cols_partial = ma.compress_cols(arr_partial)
print(f"compress_cols (partial mask) shape: {result_cols_partial.shape}")
```

**Output:**
```
compress_rows shape: (0,)
Expected: (0, 3), Actual: (0,)
compress_cols shape: (0,)
Expected: (2, 0), Actual: (0,)
compress_rows (partial mask) shape: (0, 3)
compress_cols (partial mask) shape: (2, 1)
```

## Why This Is A Bug

The functions should always preserve dimensionality. When given a 2-D array, they should return a 2-D array, even if one dimension becomes 0.

The inconsistency breaks the expected contract:
- `compress_rows` should return shape `(num_unmasked_rows, original_cols)`
- `compress_cols` should return shape `(original_rows, num_unmasked_cols)`

The current behavior causes:
1. **Shape inconsistency**: Same function returns different dimensionality based on masking pattern
2. **Downstream failures**: Code expecting 2-D arrays crashes with `IndexError` accessing `.shape[1]`
3. **Type confusion**: Return type changes from 2-D to 1-D array unexpectedly

## Fix

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -944,7 +944,11 @@ def compress_rows(x):
         The compressed array.
     """
     x = asarray(x)
-    return compress_rowcols(x, 0)
+    result = compress_rowcols(x, 0)
+    if result.size == 0 and x.ndim == 2:
+        result = result.reshape((0, x.shape[1]))
+    return result
+

 def compress_cols(x):
     """
@@ -980,7 +984,10 @@ def compress_cols(x):
         The compressed array.
     """
     x = asarray(x)
-    return compress_rowcols(x, 1)
+    result = compress_rowcols(x, 1)
+    if result.size == 0 and x.ndim == 2:
+        result = result.reshape((x.shape[0], 0))
+    return result


 def dot(a, b, strict=False, out=None):
```