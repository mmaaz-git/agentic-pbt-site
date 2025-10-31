# Bug Report: xarray.core.duck_array_ops cumprod/cumsum axis=None

**Target**: `xarray.core.duck_array_ops.cumprod` and `xarray.core.duck_array_ops.cumsum`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `axis=None`, xarray's `cumprod()` and `cumsum()` return incorrect shape and values compared to numpy. Instead of flattening the array and computing cumulative operations on the flattened 1D result (like numpy does), xarray applies the operation sequentially along each axis, producing wrong results.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from xarray.core import duck_array_ops


@given(
    st.integers(min_value=2, max_value=8),
    st.integers(min_value=2, max_value=8)
)
@settings(max_examples=100)
def test_cumprod_axis_none_matches_numpy(rows, cols):
    values = np.random.randn(rows, cols)

    xr_result = duck_array_ops.cumprod(values, axis=None)
    np_result = np.cumprod(values, axis=None)

    assert xr_result.shape == np_result.shape, \
        f"Shape mismatch: {xr_result.shape} != {np_result.shape}"
    assert np.allclose(xr_result.flatten(), np_result), \
        f"Values mismatch"
```

**Failing input**: `rows=2, cols=2` (actually fails for any multi-dimensional array)

## Reproducing the Bug

```python
import numpy as np
from xarray.core import duck_array_ops

arr = np.array([[1, 2], [3, 4]])

xarray_cumprod = duck_array_ops.cumprod(arr, axis=None)
numpy_cumprod = np.cumprod(arr, axis=None)

print(f"xarray result: shape={xarray_cumprod.shape}, values={xarray_cumprod.flatten().tolist()}")
print(f"numpy result:  shape={numpy_cumprod.shape}, values={numpy_cumprod.tolist()}")

xarray_cumsum = duck_array_ops.cumsum(arr, axis=None)
numpy_cumsum = np.cumsum(arr, axis=None)

print(f"\nxarray cumsum: shape={xarray_cumsum.shape}, values={xarray_cumsum.flatten().tolist()}")
print(f"numpy cumsum:  shape={numpy_cumsum.shape}, values={numpy_cumsum.tolist()}")
```

**Expected output**:
```
xarray result: shape=(4,), values=[1, 2, 6, 24]
numpy result:  shape=(4,), values=[1, 2, 6, 24]

xarray cumsum: shape=(4,), values=[1, 3, 6, 10]
numpy cumsum:  shape=(4,), values=[1, 3, 6, 10]
```

**Actual output**:
```
xarray result: shape=(2, 2), values=[1, 2, 3, 24]
numpy result:  shape=(4,), values=[1, 2, 6, 24]

xarray cumsum: shape=(2, 2), values=[1, 3, 4, 10]
numpy cumsum:  shape=(4,), values=[1, 3, 6, 10]
```

## Why This Is A Bug

According to numpy's documentation and standard behavior, when `axis=None` is passed to cumulative operations like `cumprod` and `cumsum`, the array should be flattened first, then the cumulative operation applied along the flattened array, returning a 1D result.

xarray's implementation in `_nd_cum_func` incorrectly treats `axis=None` as "apply to all axes sequentially" (lines 785-786 in duck_array_ops.py):

```python
if axis is None:
    axis = tuple(range(array.ndim))
```

This causes it to apply cumprod along axis 0, then along axis 1, which produces entirely different (and incorrect) results.

## Fix

```diff
--- a/xarray/core/duck_array_ops.py
+++ b/xarray/core/duck_array_ops.py
@@ -783,6 +783,8 @@ def _mean(array, axis=None, skipna=None, **kwargs):
 def _nd_cum_func(cum_func, array, axis, **kwargs):
     array = asarray(array)
     if axis is None:
+        # Match numpy behavior: flatten array first
+        return cum_func(array.flatten(), axis=0, **kwargs)
-        axis = tuple(range(array.ndim))
     if isinstance(axis, int):
         axis = (axis,)
```

Alternatively, remove the special handling of `axis=None` entirely and let the underlying `cumprod_1d`/`cumsum_1d` functions handle it naturally (they delegate to numpy/bottleneck which already handle `axis=None` correctly).