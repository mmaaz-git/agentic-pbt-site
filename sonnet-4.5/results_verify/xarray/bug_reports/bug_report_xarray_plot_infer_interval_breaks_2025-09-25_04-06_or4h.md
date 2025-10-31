# Bug Report: xarray.plot._infer_interval_breaks Incorrect Validation for Decreasing Arrays

**Target**: `xarray.plot.utils._infer_interval_breaks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_infer_interval_breaks` function with `check_monotonic=True` fails to validate that coordinates are in increasing order, accepting decreasing arrays and producing incorrect interval breaks.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from xarray.plot.utils import _infer_interval_breaks

@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=50
    )
)
def test_infer_interval_breaks_bounds(arr):
    arr = np.array(arr)
    breaks = _infer_interval_breaks(arr)
    assert breaks[0] <= arr.min(), f"First break {breaks[0]} should be <= min {arr.min()}"
    assert breaks[-1] >= arr.max(), f"Last break {breaks[-1]} should be >= max {arr.max()}"
```

**Failing input**: `[1.0, 0.0]`

## Reproducing the Bug

```python
import numpy as np
from xarray.plot.utils import _infer_interval_breaks, _is_monotonic

arr = np.array([2.0, 1.0, 0.0])

print(f"Is monotonic: {_is_monotonic(arr)}")

breaks = _infer_interval_breaks(arr, check_monotonic=True)
print(f"Breaks: {breaks}")

print(f"First break: {breaks[0]}, should encompass min: {arr.min()}")
print(f"Last break: {breaks[-1]}, should encompass max: {arr.max()}")
```

Output:
```
Is monotonic: True
Breaks: [ 2.5  1.5  0.5 -0.5]
First break: 2.5, should encompass min: 0.0
Last break: -0.5, should encompass max: 2.0
```

## Why This Is A Bug

1. The error message in `_infer_interval_breaks` explicitly states it checks for "sorted in **increasing** order"
2. However, `_is_monotonic` returns `True` for both increasing AND decreasing arrays
3. When `check_monotonic=True` is used with a decreasing array:
   - No error is raised (incorrect validation)
   - The computed interval breaks are wrong (first break > max value, last break < min value)
4. This violates the function's documented contract and produces incorrect plotting coordinates

## Fix

The `_is_monotonic` function should check for increasing order when used in `_infer_interval_breaks`, or `_infer_interval_breaks` should use a different validation function.

```diff
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -831,7 +831,7 @@ def _update_axes(
 def _is_monotonic(coord, axis=0):
     """
-    >>> _is_monotonic(np.array([0, 1, 2]))
+    Check if array is monotonic (increasing or decreasing)
     np.True_
     >>> _is_monotonic(np.array([2, 1, 0]))
     np.True_
@@ -848,6 +848,17 @@ def _is_monotonic(coord, axis=0):
             np.arange(0, n - 1), axis=axis
         )
         return np.all(delta_pos) or np.all(delta_neg)
+
+
+def _is_monotonic_increasing(coord, axis=0):
+    """Check if array is monotonic increasing"""
+    if coord.shape[axis] < 2:
+        return True
+    else:
+        n = coord.shape[axis]
+        delta_pos = coord.take(np.arange(1, n), axis=axis) >= coord.take(
+            np.arange(0, n - 1), axis=axis
+        )
+        return np.all(delta_pos)


 def _infer_interval_breaks(coord, axis=0, scale=None, check_monotonic=False):
@@ -863,7 +874,7 @@ def _infer_interval_breaks(coord, axis=0, scale=None, check_monotonic=False):
     """
     coord = np.asarray(coord)

-    if check_monotonic and not _is_monotonic(coord, axis=axis):
+    if check_monotonic and not _is_monotonic_increasing(coord, axis=axis):
         raise ValueError(
             "The input coordinate is not sorted in increasing "
             f"order along axis {axis}. This can lead to unexpected "