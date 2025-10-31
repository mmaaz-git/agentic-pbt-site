# Bug Report: xarray.plot.utils._rescale_imshow_rgb Division by Zero with Constant Arrays

**Target**: `xarray.plot.utils._rescale_imshow_rgb`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_rescale_imshow_rgb` function produces NaN values when processing constant arrays (where all values are identical) with `robust=True`, due to division by zero when the computed vmin equals vmax.

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

if __name__ == "__main__":
    test_rescale_imshow_rgb_robust()
```

<details>

<summary>
**Failing input**: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
</summary>
```
/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/plot/utils.py:778: RuntimeWarning: invalid value encountered in divide
  darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 17, in <module>
    test_rescale_imshow_rgb_robust()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 9, in test_rescale_imshow_rgb_robust
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 13, in test_rescale_imshow_rgb_robust
    assert np.all(result >= 0)
           ~~~~~~^^^^^^^^^^^^^
AssertionError
Falsifying example: test_rescale_imshow_rgb_robust(
    values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb

# Test with constant array of zeros
print("Test 1: Constant array of zeros")
darray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
print(f"Input: {darray}")
print(f"Output: {result}")
print(f"Contains NaN: {np.any(np.isnan(result))}")
print()

# Test with constant array of fives
print("Test 2: Constant array of fives")
darray = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
print(f"Input: {darray}")
print(f"Output: {result}")
print(f"Contains NaN: {np.any(np.isnan(result))}")
print()

# Test with constant array of 255 (typical for image data)
print("Test 3: Constant array of 255 (white image)")
darray = np.array([255.0, 255.0, 255.0, 255.0, 255.0, 255.0])
result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
print(f"Input: {darray}")
print(f"Output: {result}")
print(f"Contains NaN: {np.any(np.isnan(result))}")
```

<details>

<summary>
RuntimeWarning: Division by zero producing NaN values
</summary>
```
/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/plot/utils.py:778: RuntimeWarning: invalid value encountered in divide
  darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
Test 1: Constant array of zeros
Input: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Output: [nan nan nan nan nan nan nan nan nan nan]
Contains NaN: True

Test 2: Constant array of fives
Input: [5. 5. 5. 5. 5.]
Output: [nan nan nan nan nan]
Contains NaN: True

Test 3: Constant array of 255 (white image)
Input: [255. 255. 255. 255. 255. 255.]
Output: [nan nan nan nan nan nan]
Contains NaN: True
```
</details>

## Why This Is A Bug

This violates the function's expected behavior in multiple ways:

1. **Contract Violation**: The function ends with `return np.minimum(np.maximum(darray, 0), 1)` (line 779), clearly intending to return values bounded to [0, 1]. However, NaN values bypass these bounds since NaN comparisons always return False, violating the function's implicit contract.

2. **Mathematical Undefined Behavior**: When `robust=True`, the function uses `np.nanpercentile` with `ROBUST_PERCENTILE=2.0` to compute vmin and vmax. For constant arrays, both the 2nd and 98th percentiles equal the constant value, making vmin == vmax. The scaling formula at line 778 then performs `(darray - vmin) / (vmax - vmin)`, which is division by zero.

3. **Silent Data Corruption**: The function returns NaN values that propagate silently through subsequent computations rather than raising an explicit error, making debugging difficult for users.

4. **Documentation Mismatch**: The function is designed to rescale image RGB values for display, as indicated by its name and use in the plotting module. Constant-valued images (solid colors) are valid and common in image processing, yet the function fails on these legitimate inputs.

## Relevant Context

The function is located in `/xarray/plot/utils.py` and is used internally by xarray's plotting functionality to normalize image data for matplotlib display. The `ROBUST_PERCENTILE` constant is defined as 2.0 at line 54 of the same file.

Common real-world scenarios where this bug manifests:
- Solid color images (all black, all white, or any uniform color)
- Uniform regions in scientific data visualizations
- Placeholder images or backgrounds
- Medical imaging data with uniform regions
- Satellite imagery with uniform areas (water, clouds)

Similar functions in the ecosystem handle this case gracefully. For example, matplotlib's `Normalize` class returns 0.0 for constant arrays when vmin equals vmax, avoiding the division by zero.

The function already has error handling for invalid vmin/vmax combinations (lines 762-773), but lacks protection against the equal values case that occurs naturally with constant input data.

## Proposed Fix

```diff
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -772,6 +772,11 @@ def _rescale_imshow_rgb(darray, vmin, vmax, robust):
                 "a vmin < vmax in this case."
             )

+    # Handle constant arrays where vmin == vmax
+    if vmin == vmax:
+        # For constant arrays, return 0.5 (middle of [0, 1] range)
+        return np.full(darray.shape, 0.5, dtype="f4")
+
     # Scale interval [vmin .. vmax] to [0 .. 1], with darray as 64-bit float
     # to avoid precision loss, integer over/underflow, etc with extreme inputs.
     # After scaling, downcast to 32-bit float.  This substantially reduces
```