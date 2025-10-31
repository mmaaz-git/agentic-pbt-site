# Bug Report: xarray.plot._rescale_imshow_rgb Division by Zero with Constant Data

**Target**: `xarray.plot.utils._rescale_imshow_rgb`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_rescale_imshow_rgb` function crashes with a division-by-zero error (producing NaN values) when given constant data (all values the same) with `robust=True` or when `vmin == vmax`. This can occur in realistic scenarios such as plotting RGB images with uniform color channels.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st
from xarray.plot.utils import _rescale_imshow_rgb


@given(
    st.lists(st.floats(min_value=0, max_value=255, allow_nan=False, allow_infinity=False), min_size=10, max_size=100)
)
def test_rescale_imshow_rgb_robust(data):
    darray = np.array(data)
    result = _rescale_imshow_rgb(darray, None, None, True)
    assert np.all(result >= 0)
    assert np.all(result <= 1)
    assert np.all(np.isfinite(result))
```

**Failing input**: `data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb

constant_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
result = _rescale_imshow_rgb(constant_data, None, None, robust=True)

print(f"Input: {constant_data}")
print(f"Result: {result}")
print(f"Contains NaN: {np.any(np.isnan(result))}")

constant_data2 = np.array([5.0, 5.0, 5.0])
result2 = _rescale_imshow_rgb(constant_data2, vmin=5.0, vmax=5.0, robust=False)

print(f"\nInput: {constant_data2}")
print(f"Result: {result2}")
print(f"Contains NaN: {np.any(np.isnan(result2))}")
```

## Why This Is A Bug

When all data values are constant:
1. With `robust=True`, `vmin` and `vmax` are computed as percentiles, which will be equal
2. When explicitly passed `vmin == vmax`
3. Line 778 performs division by `(vmax - vmin)`, which equals zero
4. This produces NaN values instead of valid normalized data
5. The function has validation for `vmin < vmax` in some code paths (lines 762-773) but NOT for the robust case

This violates the function's implicit contract of returning valid normalized data in the range [0, 1]. The bug can occur in realistic scenarios when plotting RGB images with uniform color channels (e.g., a completely red, green, or blue image).

## Fix

The function should handle the case where `vmin == vmax` by returning a constant value (typically 0.5 or the data itself if it's already in [0, 1]). Here's a patch:

```diff
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -771,6 +771,12 @@ def _rescale_imshow_rgb(darray, vmin, vmax, robust):
             raise ValueError(
                 f"vmax={vmax!r} is less than the default vmin (0) - you must supply "
                 "a vmin < vmax in this case."
             )
+
+    # Handle the case where vmin == vmax (constant data)
+    if vmin == vmax:
+        # Return constant value of 0.5 when data is uniform
+        return np.full_like(darray, 0.5, dtype="f4")
+
     # Scale interval [vmin .. vmax] to [0 .. 1], with darray as 64-bit float
     # to avoid precision loss, integer over/underflow, etc with extreme inputs.
     # After scaling, downcast to 32-bit float.  This substantially reduces
```

Alternative fix (more conservative, just clip to avoid NaN):

```diff
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -774,7 +774,10 @@ def _rescale_imshow_rgb(darray, vmin, vmax, robust):
     # Scale interval [vmin .. vmax] to [0 .. 1], with darray as 64-bit float
     # to avoid precision loss, integer over/underflow, etc with extreme inputs.
     # After scaling, downcast to 32-bit float.  This substantially reduces
     # memory usage after we hand `darray` off to matplotlib.
-    darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
+    denominator = vmax - vmin
+    if denominator == 0:
+        darray = np.full_like(darray, 0.5, dtype="f4")
+    else:
+        darray = ((darray.astype("f8") - vmin) / denominator).astype("f4")
     return np.minimum(np.maximum(darray, 0), 1)
```