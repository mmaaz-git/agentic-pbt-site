# Bug Report: xarray.plot.utils._infer_interval_breaks Incorrect Results for Non-Monotonic Data

**Target**: `xarray.plot.utils._infer_interval_breaks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_infer_interval_breaks` function produces incorrect interval breaks for non-monotonic coordinate data, resulting in breaks that don't contain all the original coordinate values. This can cause plotting functions like `pcolormesh` to display data incorrectly.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import xarray.plot.utils as plot_utils

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            min_size=2, max_size=20)
)
def test_infer_interval_breaks_contains_original(values):
    """Property: interval breaks should contain all original points"""
    arr = np.array(values)
    breaks = plot_utils._infer_interval_breaks(arr)

    for i in range(len(arr)):
        min_break = min(breaks)
        max_break = max(breaks)
        assert min_break <= arr[i] <= max_break, \
            f"Original value {arr[i]} not in range [{min_break}, {max_break}]"
```

**Failing input**: `[0.0, 1.0, 0.0]`

## Reproducing the Bug

```python
import numpy as np
import xarray.plot.utils as plot_utils

coord = np.array([0.0, 1.0, 0.0])
breaks = plot_utils._infer_interval_breaks(coord)

print(f"Input coordinates: {coord}")
print(f"Inferred breaks:   {breaks}")
print(f"Breaks range: [{min(breaks)}, {max(breaks)}]")

assert all(min(breaks) <= c <= max(breaks) for c in coord)
```

Output:
```
Input coordinates: [0. 1. 0.]
Inferred breaks:   [-0.5  0.5  0.5 -0.5]
Breaks range: [-0.5, 0.5]
AssertionError: coordinate value 1.0 not in range [-0.5, 0.5]
```

## Why This Is A Bug

The function computes interval breaks using local differences (`0.5 * np.diff(coord)`). For non-monotonic data like `[0.0, 1.0, 0.0]`:
- deltas = `[0.5, -0.5]`
- breaks = `[-0.5, 0.5, 0.5, -0.5]`
- range: `[-0.5, 0.5]`

This fails to contain the middle value `1.0`, which violates the fundamental property that interval breaks should encompass all coordinate values for proper visualization.

While the function has a `check_monotonic` parameter, it defaults to `False`, so users receive no warning when passing non-monotonic data. The documentation only shows monotonic examples and doesn't specify that monotonic input is required.

## Fix

The function should either:

1. **Default `check_monotonic=True`** to warn users about non-monotonic data
2. **Handle non-monotonic data correctly** by ensuring breaks always encompass all values
3. **Document the monotonic requirement** clearly in the docstring

Option 1 (simplest fix):

```diff
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -100,7 +100,7 @@ def _infer_interval_breaks(coord, axis=0, scale=None, check_monotonic=False):
-def _infer_interval_breaks(coord, axis=0, scale=None, check_monotonic=False):
+def _infer_interval_breaks(coord, axis=0, scale=None, check_monotonic=True):
     """
     >>> _infer_interval_breaks(np.arange(5))
     array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])
```

This would make the function raise a clear error when given non-monotonic data, preventing silent incorrect results.