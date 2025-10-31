# Bug Report: xarray.plot.utils._rescale_imshow_rgb Division by Zero

**Target**: `xarray.plot.utils._rescale_imshow_rgb`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_rescale_imshow_rgb` function produces NaN values when all input values are equal (constant array) in robust mode, due to division by zero when `vmin == vmax`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.plot.utils import _rescale_imshow_rgb

@given(st.lists(st.floats(min_value=0, max_value=255, allow_nan=False, allow_infinity=False), min_size=10, max_size=100))
@settings(max_examples=500)
def test_rescale_imshow_rgb_robust(values):
    darray = np.array(values)
    result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
    assert np.all(result >= 0)
    assert np.all(result <= 1)
```

**Failing input**: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`

## Reproducing the Bug

```python
import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb

darray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
print(result)
```

Output:
```
[nan nan nan nan nan nan nan nan nan nan]
```

The same issue occurs with any constant array:
```python
darray = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
print(result)
```

Output:
```
[nan nan nan nan nan]
```

## Why This Is A Bug

1. When `robust=True`, the function computes `vmin` and `vmax` using percentiles
2. For a constant array, both percentiles return the same value, so `vmin == vmax`
3. The scaling formula `(darray - vmin) / (vmax - vmin)` divides by zero
4. This produces NaN values which violate the function's contract of returning values in [0, 1]
5. This can occur in realistic scenarios like plotting a completely uniform image (e.g., all black, all white, or any constant color)

## Fix

```diff
diff --git a/xarray/plot/utils.py b/xarray/plot/utils.py
index 1234567..abcdefg 100644
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -772,6 +772,12 @@ def _rescale_imshow_rgb(darray, vmin, vmax, robust):
                 f"vmax={vmax!r} is less than the default vmin (0) - you must supply "
                 "a vmin < vmax in this case."
             )
+
+    # Handle constant arrays where vmin == vmax
+    if vmin == vmax:
+        # For constant arrays, all values map to the middle of [0, 1]
+        return np.full(darray.shape, 0.5, dtype="f4")
+
     # Scale interval [vmin .. vmax] to [0 .. 1], with darray as 64-bit float
     # to avoid precision loss, integer over/underflow, etc with extreme inputs.
     # After scaling, downcast to 32-bit float.  This substantially reduces
```