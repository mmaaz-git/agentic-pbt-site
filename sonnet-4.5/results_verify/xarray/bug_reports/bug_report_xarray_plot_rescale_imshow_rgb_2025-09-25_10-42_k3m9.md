# Bug Report: xarray.plot.utils._rescale_imshow_rgb Division by Zero

**Target**: `xarray.plot.utils._rescale_imshow_rgb`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_rescale_imshow_rgb` function produces NaN values when vmin equals vmax (e.g., when rescaling a constant array), violating its postcondition that output values must be in the [0, 1] range.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.plot.utils import _rescale_imshow_rgb


@given(
    arr=st.lists(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False), min_size=1, max_size=100),
)
@settings(max_examples=1000)
def test_rescale_imshow_rgb_with_robust(arr):
    darray = np.array(arr)
    result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
    assert np.all(result >= 0), f"Result has values < 0: {result}"
    assert np.all(result <= 1), f"Result has values > 1: {result}"
```

**Failing input**: `arr=[0.0]`

## Reproducing the Bug

```python
import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb

darray = np.array([0.0])
result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
print(f"Result: {result}")
print(f"Contains NaN: {np.any(np.isnan(result))}")

darray2 = np.array([5.0, 5.0, 5.0])
result2 = _rescale_imshow_rgb(darray2, vmin=None, vmax=None, robust=True)
print(f"Result: {result2}")
print(f"Contains NaN: {np.any(np.isnan(result2))}")

darray3 = np.array([1.0, 2.0, 3.0])
result3 = _rescale_imshow_rgb(darray3, vmin=2.0, vmax=2.0, robust=False)
print(f"Result: {result3}")
print(f"Contains NaN: {np.any(np.isnan(result3))}")
```

## Why This Is A Bug

When the input array is constant (all values the same), or when vmin is explicitly set equal to vmax:

1. With `robust=True` and constant array `[0.0]`:
   - `vmin = np.nanpercentile([0.0], 2.0) = 0.0`
   - `vmax = np.nanpercentile([0.0], 98.0) = 0.0`

2. The scaling formula at line 778:
   ```python
   darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
   ```
   becomes: `(0.0 - 0.0) / (0.0 - 0.0) = 0.0 / 0.0 = NaN`

3. The function returns `np.minimum(np.maximum(NaN, 0), 1) = NaN`

The function's implicit postcondition (output in [0, 1]) is violated. Additionally, the code comment at line 774-777 states the purpose is to "Scale interval [vmin .. vmax] to [0 .. 1]", which cannot produce NaN.

## Fix

The function should handle the case when `vmin == vmax` by returning a constant array (e.g., 0.5 or 0.0) instead of performing division by zero:

```diff
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -774,7 +774,12 @@ def _rescale_imshow_rgb(darray, vmin, vmax, robust):
     # Scale interval [vmin .. vmax] to [0 .. 1], with darray as 64-bit float
     # to avoid precision loss, integer over/underflow, etc with extreme inputs.
     # After scaling, downcast to 32-bit float.  This substantially reduces
     # memory usage after we hand `darray` off to matplotlib.
-    darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
+    if vmax == vmin:
+        # Constant value case: map to middle of [0, 1] range
+        darray = np.full_like(darray, 0.5, dtype="f4")
+    else:
+        darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
     return np.minimum(np.maximum(darray, 0), 1)
```

Alternatively, following matplotlib's convention for constant colormaps, the function could map constant values to 0.0 or handle them explicitly upstream.