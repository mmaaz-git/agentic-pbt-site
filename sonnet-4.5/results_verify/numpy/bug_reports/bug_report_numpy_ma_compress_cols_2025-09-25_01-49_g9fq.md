# Bug Report: numpy.ma.compress_cols Dimensionality Loss

**Target**: `numpy.ma.compress_cols`, `numpy.ma.compress_rows`, `numpy.ma.compress_nd`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When all rows/columns are removed, `compress_cols`, `compress_rows`, and `compress_nd` return 1D arrays instead of preserving the 2D structure, violating the documented contract that these functions operate on 2D arrays.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

@given(
    arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 10), st.integers(1, 10)))
)
def test_compress_cols_removes_masked(data):
    mask = np.zeros(data.shape, dtype=bool)
    if data.shape[0] > 0 and data.shape[1] > 0:
        mask[0, 0] = True

    masked_arr = ma.array(data, mask=mask)
    result = ma.compress_cols(masked_arr)

    if mask[0, 0]:
        assert result.shape[1] < data.shape[1]
```

**Failing input**: `data=array([[0.]])`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

data = np.array([[0.]])
mask = np.array([[True]])
masked_arr = ma.array(data, mask=mask)

result = ma.compress_cols(masked_arr)

print(f"Input shape: {data.shape}, ndim: {data.ndim}")
print(f"Result shape: {result.shape}, ndim: {result.ndim}")
```

Output:
```
Input shape: (1, 1), ndim: 2
Result shape: (0,), ndim: 1
```

## Why This Is A Bug

The documentation states these functions work on "2-D arrays" and return "compressed_array : ndarray". When all columns are removed from a `(m, n)` array, the natural expectation is a `(m, 0)` shaped result, not `(0,)`. This breaks:

1. **Dimensionality invariant**: 2D input should produce 2D output
2. **User code**: Accessing `result.shape[1]` after `compress_cols` crashes with IndexError
3. **Consistency**: Partial removal preserves 2D shape, but complete removal doesn't

## Fix

The bug is in `compress_nd` in `numpy/ma/extras.py`:

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -897,7 +897,11 @@ def compress_nd(x, axis=None):
     if m is nomask or not m.any():
         return x._data
     # All is masked: return empty
     if m.all():
-        return nxarray([])
+        new_shape = list(x.shape)
+        for ax in axis:
+            new_shape[ax] = 0
+        return nxarray([]).reshape(new_shape)
     # Filter elements through boolean indexing
     data = x._data
```