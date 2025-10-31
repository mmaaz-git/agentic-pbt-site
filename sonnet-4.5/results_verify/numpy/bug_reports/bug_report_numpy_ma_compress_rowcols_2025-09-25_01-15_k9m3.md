# Bug Report: numpy.ma.compress_rowcols Dimensionality Loss

**Target**: `numpy.ma.compress_rowcols`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.ma.compress_rowcols` returns a 1D empty array when all rows/columns are masked, but returns a 2D array when only some are masked. This inconsistent dimensionality can break downstream code expecting consistent array shapes.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings

@settings(max_examples=300)
@given(
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=2, max_value=10)
)
def test_compress_rowcols_ndim_consistency(rows, cols):
    arr = np.arange(rows * cols).reshape(rows, cols).astype(float)
    all_masked = ma.array(arr, mask=np.ones((rows, cols), dtype=bool))

    result = ma.compress_rowcols(all_masked, 0)

    assert result.ndim == 2, f"Expected 2D array, got {result.ndim}D with shape {result.shape}"
```

**Failing input**: `rows=2, cols=2` (or any valid 2D dimensions)

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

arr = np.ones((3, 4))
all_masked = ma.array(arr, mask=np.ones((3, 4), dtype=bool))

result = ma.compress_rowcols(all_masked, axis=0)

print(f"Input shape: {all_masked.shape}")
print(f"Result shape: {result.shape}")
print(f"Result ndim: {result.ndim}")

assert result.ndim == 2, f"Expected 2D, got {result.ndim}D"
```

Output:
```
Input shape: (3, 4)
Result shape: (0,)
Result ndim: 1
AssertionError: Expected 2D, got 1D
```

## Why This Is A Bug

1. **Inconsistent behavior**: When removing SOME rows, returns shape `(n, cols)`. When removing ALL rows, returns shape `(0,)` instead of `(0, cols)`.

2. **Breaks expectations**: The function takes a 2D array and should return a 2D array. Users expect dimensionality to be preserved.

3. **Documented as 2D-only**: The docstring states "Must be a 2D array" for input but doesn't document returning 1D output.

4. **Violates the property**: For 2D input with shape `(r, c)`, compressing along axis 0 should yield shape `(r', c)` where `0 <= r' <= r`, not shape `(r',)`.

## Fix

The bug is in `compress_nd` (called by `compress_rowcols`):

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -XXX,8 +XXX,17 @@ def compress_nd(x, axis=None):
     # Nothing is masked: return x
     if m is nomask or not m.any():
         return x._data
-    # All is masked: return empty
+    # All is masked: return appropriately shaped empty array
     if m.all():
-        return nxarray([])
+        # Compute the shape with compressed dimensions set to 0
+        new_shape = list(x.shape)
+        for ax in axis:
+            new_shape[ax] = 0
+        return nxarray([]).reshape(new_shape)
     # Filter elements through boolean indexing
     data = x._data
```

This ensures that when all elements along specified axes are masked, the result maintains the correct dimensionality with size 0 along the compressed axes.