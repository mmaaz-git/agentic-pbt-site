# Bug Report: xarray.plot.utils._rescale_imshow_rgb Division by Zero with Constant Arrays

**Target**: `xarray.plot.utils._rescale_imshow_rgb`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_rescale_imshow_rgb` function produces NaN values due to division by zero when processing constant arrays (where all values are identical) or when vmin equals vmax, violating its postcondition that output values must be in the [0, 1] range.

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

if __name__ == "__main__":
    test_rescale_imshow_rgb_with_robust()
```

<details>

<summary>
**Failing input**: `arr=[0.0]`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/xarray/plot/utils.py:778: RuntimeWarning: invalid value encountered in divide
  darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 17, in <module>
    test_rescale_imshow_rgb_with_robust()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 7, in test_rescale_imshow_rgb_with_robust
    arr=st.lists(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False), min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 13, in test_rescale_imshow_rgb_with_robust
    assert np.all(result >= 0), f"Result has values < 0: {result}"
           ~~~~~~^^^^^^^^^^^^^
AssertionError: Result has values < 0: [nan]
Falsifying example: test_rescale_imshow_rgb_with_robust(
    arr=[0.0],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1085
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1708
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_function_base_impl.py:4768
```
</details>

## Reproducing the Bug

```python
import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb

# Test case 1: Single constant value
print("Test 1: Single constant value [0.0]")
darray = np.array([0.0])
result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
print(f"Input: {darray}")
print(f"Result: {result}")
print(f"Contains NaN: {np.any(np.isnan(result))}")
print()

# Test case 2: Multiple identical values
print("Test 2: Multiple identical values [5.0, 5.0, 5.0]")
darray2 = np.array([5.0, 5.0, 5.0])
result2 = _rescale_imshow_rgb(darray2, vmin=None, vmax=None, robust=True)
print(f"Input: {darray2}")
print(f"Result: {result2}")
print(f"Contains NaN: {np.any(np.isnan(result2))}")
print()

# Test case 3: Explicitly set vmin=vmax
print("Test 3: Explicitly set vmin=vmax with [1.0, 2.0, 3.0]")
darray3 = np.array([1.0, 2.0, 3.0])
result3 = _rescale_imshow_rgb(darray3, vmin=2.0, vmax=2.0, robust=False)
print(f"Input: {darray3}")
print(f"Result: {result3}")
print(f"Contains NaN: {np.any(np.isnan(result3))}")
```

<details>

<summary>
RuntimeWarning: divide by zero and invalid value encountered - produces NaN values
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/xarray/plot/utils.py:778: RuntimeWarning: invalid value encountered in divide
  darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
/home/npc/miniconda/lib/python3.13/site-packages/xarray/plot/utils.py:778: RuntimeWarning: divide by zero encountered in divide
  darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
Test 1: Single constant value [0.0]
Input: [0.]
Result: [nan]
Contains NaN: True

Test 2: Multiple identical values [5.0, 5.0, 5.0]
Input: [5. 5. 5.]
Result: [nan nan nan]
Contains NaN: True

Test 3: Explicitly set vmin=vmax with [1.0, 2.0, 3.0]
Input: [1. 2. 3.]
Result: [ 0. nan  1.]
Contains NaN: True
```
</details>

## Why This Is A Bug

This function violates its documented purpose and postcondition in several ways:

1. **Division by Zero**: When the input array contains all identical values (constant array), the function computes vmin and vmax using percentiles. For a constant array, both percentiles return the same value, resulting in `vmax - vmin = 0`. The scaling formula at line 778 then performs division by zero: `(value - vmin) / 0 = NaN`.

2. **Violated Contract**: The function's comment at lines 774-777 explicitly states its purpose is to "Scale interval [vmin .. vmax] to [0 .. 1]". The function returns values clamped to [0, 1] using `np.minimum(np.maximum(darray, 0), 1)` at line 779, but NaN values pass through these operations unchanged, violating the [0, 1] range guarantee.

3. **Common Use Case**: Constant arrays are legitimate inputs in visualization contexts, such as displaying solid colors, uniform backgrounds, or data that happens to have no variation. The function should handle these gracefully rather than producing invalid output.

4. **Silent Failure**: The function produces NaN values that can propagate through visualization pipelines, causing mysterious rendering issues or crashes in downstream code without clear error messages.

## Relevant Context

- The function is used internally by xarray's plotting utilities, particularly for RGB image display via `imshow`
- The `ROBUST_PERCENTILE` constant is set to 2.0, meaning robust mode uses the 2nd and 98th percentiles
- The function already handles edge cases where vmin > vmax by raising ValueError, but doesn't handle vmin == vmax
- NumPy's behavior with NaN: `np.minimum(NaN, x) = NaN` and `np.maximum(NaN, x) = NaN`, so the clamping operation doesn't fix NaN values
- Similar functions in matplotlib and other visualization libraries typically map constant values to a specific value (often 0.0 or 0.5) to avoid this issue

Documentation reference: The function is part of xarray's internal plotting utilities and is called by public-facing functions like `DataArray.plot.imshow()` when displaying RGB images.

## Proposed Fix

```diff
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -773,6 +773,11 @@ def _rescale_imshow_rgb(darray, vmin, vmax, robust):
             )
+    # Handle case where vmin equals vmax (constant array)
+    if vmin == vmax:
+        # Map constant values to middle of [0, 1] range
+        return np.full_like(darray, 0.5, dtype="f4")
+
     # Scale interval [vmin .. vmax] to [0 .. 1], with darray as 64-bit float
     # to avoid precision loss, integer over/underflow, etc with extreme inputs.
     # After scaling, downcast to 32-bit float.  This substantially reduces
     # memory usage after we hand `darray` off to matplotlib.
     darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
     return np.minimum(np.maximum(darray, 0), 1)
```