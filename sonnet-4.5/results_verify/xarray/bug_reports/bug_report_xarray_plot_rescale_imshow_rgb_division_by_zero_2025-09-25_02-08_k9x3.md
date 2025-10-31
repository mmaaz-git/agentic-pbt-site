# Bug Report: xarray.plot.utils._rescale_imshow_rgb Division by Zero

**Target**: `xarray.plot.utils._rescale_imshow_rgb`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_rescale_imshow_rgb` function produces NaN values when all input data values are identical and `robust=True`, due to division by zero when `vmax == vmin`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import xarray as xr
from xarray.plot.utils import _rescale_imshow_rgb

@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=10,
        max_size=100
    )
)
def test_rescale_imshow_rgb_output_range_robust(data_list):
    data_values = np.array(data_list).reshape(-1, 1, 1)
    darray = xr.DataArray(data_values)

    result = _rescale_imshow_rgb(darray, None, None, robust=True)

    assert np.all(result >= 0), f"Found values < 0: min={result.min()}"
    assert np.all(result <= 1), f"Found values > 1: max={result.max()}"
```

**Failing input**: `data_list=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`

## Reproducing the Bug

```python
import numpy as np
import xarray as xr
from xarray.plot.utils import _rescale_imshow_rgb

data_values = np.array([0.0] * 10).reshape(-1, 1, 1)
darray = xr.DataArray(data_values)

result = _rescale_imshow_rgb(darray, None, None, robust=True)

print(f"Output contains NaN: {np.any(np.isnan(result.values))}")
```

Output:
```
Output contains NaN: True
```

## Why This Is A Bug

When `robust=True` and all data values are identical, the function calculates:
- `vmin = np.nanpercentile(data, 2.0)` → value
- `vmax = np.nanpercentile(data, 98.0)` → same value

Then at line 778, the function performs:
```python
darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
```

Since `vmax - vmin = 0`, this results in division by zero, producing NaN values. The function's postcondition (returning values in [0, 1]) is violated.

This affects users when plotting uniform-valued data with `robust=True`, producing invalid visualizations.

## Fix

```diff
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -756,6 +756,11 @@ def _rescale_imshow_rgb(darray, vmin, vmax, robust):
         if vmin is None:
             vmin = np.nanpercentile(darray, ROBUST_PERCENTILE)
+    # Handle case where all values are identical (vmax == vmin)
+    if vmax == vmin:
+        # Return array of 0.5 (middle of [0, 1] range) when data has no variation
+        return np.full_like(darray, 0.5, dtype="f4")
+
     # If not robust and one bound is None, calculate the default other bound
     # and check that an interval between them exists.
     elif vmax is None:
```