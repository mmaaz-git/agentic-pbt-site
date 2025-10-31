# Bug Report: xarray.plot.utils._infer_interval_breaks Unsorted Coordinates

**Target**: `xarray.plot.utils._infer_interval_breaks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`_infer_interval_breaks` produces interval breaks that don't cover the full data range when coordinates are unsorted, leading to incorrect plot boundaries.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from xarray.plot.utils import _infer_interval_breaks

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-1e6, max_value=1e6),
               min_size=2, max_size=100))
def test_infer_interval_breaks_covers_input_range(coord):
    coord_arr = np.array(coord)
    result = _infer_interval_breaks(coord_arr)
    assert result[0] <= coord_arr.min(), \
        f"First break should be <= min, got {result[0]} > {coord_arr.min()}"
    assert result[-1] >= coord_arr.max(), \
        f"Last break should be >= max, got {result[-1]} < {coord_arr.max()}"
```

**Failing input**: `coord=[0.0, -1.0]`

## Reproducing the Bug

```python
import numpy as np
from xarray.plot.utils import _infer_interval_breaks

coord = np.array([0.0, -1.0])
result = _infer_interval_breaks(coord)

print(f"Input coordinates: {coord}")
print(f"Data range: [{coord.min()}, {coord.max()}]")
print(f"Interval breaks: {result}")
print(f"Breaks range: [{result[0]}, {result[-1]}]")
```

**Output**:
```
Input coordinates: [ 0. -1.]
Data range: [-1.0, 0.0]
Interval breaks: [ 0.5 -0.5 -1.5]
Breaks range: [0.5, -1.5]
```

**Problem**: First break (0.5) > minimum value (-1.0), and last break (-1.5) < maximum value (0.0).

## Why This Is A Bug

The function is documented to "infer interval breaks" for plotting coordinates. When used for visualization functions like `pcolormesh`, these breaks define the boundaries of plotted cells. If the breaks don't cover the full data range, some data points will fall outside the plotted region, leading to incorrect or incomplete visualizations.

The function incorrectly assumes that `coord[0]` is the minimum and `coord[-1]` is the maximum, but this is only true for sorted arrays. While there is a `check_monotonic` parameter that can warn about unsorted data, it defaults to `False`, and the function still produces incorrect results even when unsorted data is detected.

This is particularly problematic because:
1. Users may not always pass sorted coordinates
2. The function silently produces incorrect results
3. The resulting visualization errors may not be immediately obvious

## Fix

The function should ensure that interval breaks cover the actual data range regardless of sort order. Here's a minimal fix:

```diff
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -892,6 +892,14 @@ def _infer_interval_breaks(coord, axis=0, scale=None, check_monotonic=False):
     interval_breaks = np.concatenate(
         [first, coord[trim_last] + deltas, last], axis=axis
     )
+
+    # Ensure breaks cover the full data range, even for unsorted input
+    coord_min = np.min(coord, axis=axis, keepdims=True)
+    coord_max = np.max(coord, axis=axis, keepdims=True)
+    breaks_min = np.min(interval_breaks, axis=axis, keepdims=True)
+    breaks_max = np.max(interval_breaks, axis=axis, keepdims=True)
+    interval_breaks = np.where(interval_breaks == breaks_min, np.minimum(interval_breaks, coord_min - np.abs(coord_min) * 1e-10), interval_breaks)
+    interval_breaks = np.where(interval_breaks == breaks_max, np.maximum(interval_breaks, coord_max + np.abs(coord_max) * 1e-10), interval_breaks)
     if scale == "log":
         # Recovert the intervals into the linear space
         return np.power(10, interval_breaks)
```

However, a cleaner approach would be to make `check_monotonic=True` by default and raise an error, or explicitly document that coordinates must be sorted.