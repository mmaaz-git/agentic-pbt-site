# Bug Report: xarray.plot.utils._infer_interval_breaks Incorrect Interval Breaks for Non-Ascending Coordinates

**Target**: `xarray.plot.utils._infer_interval_breaks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`_infer_interval_breaks` produces mathematically incorrect interval breaks that don't cover the input data range when coordinates are not in ascending order, causing data points to fall outside the plotted region in visualization functions.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
import numpy as np
from xarray.plot.utils import _infer_interval_breaks

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-1e6, max_value=1e6),
               min_size=2, max_size=100))
@example([0.0, -1.0])  # The minimal failing case
def test_infer_interval_breaks_covers_input_range(coord):
    coord_arr = np.array(coord)
    result = _infer_interval_breaks(coord_arr)

    # Property 1: First break should be <= minimum coordinate value
    assert result[0] <= coord_arr.min(), \
        f"First break should be <= min, got {result[0]} > {coord_arr.min()}"

    # Property 2: Last break should be >= maximum coordinate value
    assert result[-1] >= coord_arr.max(), \
        f"Last break should be >= max, got {result[-1]} < {coord_arr.max()}"

if __name__ == "__main__":
    # Run the test
    test_infer_interval_breaks_covers_input_range()
```

<details>

<summary>
**Failing input**: `coord=[0.0, -1.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 23, in <module>
    test_infer_interval_breaks_covers_input_range()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 6, in test_infer_interval_breaks_covers_input_range
    min_value=-1e6, max_value=1e6),

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 14, in test_infer_interval_breaks_covers_input_range
    assert result[0] <= coord_arr.min(), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: First break should be <= min, got 0.5 > -1.0
Falsifying explicit example: test_infer_interval_breaks_covers_input_range(
    coord=[0.0, -1.0],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from xarray.plot.utils import _infer_interval_breaks

# Test case with unsorted coordinates
coord = np.array([0.0, -1.0])
result = _infer_interval_breaks(coord)

print(f"Input coordinates: {coord}")
print(f"Data range: [{coord.min()}, {coord.max()}]")
print(f"Interval breaks: {result}")
print(f"Breaks range: [{result[0]}, {result[-1]}]")
print()
print("Analysis:")
print(f"- First break ({result[0]}) should be <= minimum value ({coord.min()})")
print(f"  Result: {result[0]} <= {coord.min()} is {result[0] <= coord.min()}")
print(f"- Last break ({result[-1]}) should be >= maximum value ({coord.max()})")
print(f"  Result: {result[-1]} >= {coord.max()} is {result[-1] >= coord.max()}")
print()

# Additional test with descending sorted coordinates
coord_desc = np.array([1.0, 0.0, -1.0])
result_desc = _infer_interval_breaks(coord_desc)

print("Test with descending sorted coordinates:")
print(f"Input coordinates: {coord_desc}")
print(f"Data range: [{coord_desc.min()}, {coord_desc.max()}]")
print(f"Interval breaks: {result_desc}")
print(f"Breaks range: [{result_desc[0]}, {result_desc[-1]}]")
print(f"- First break ({result_desc[0]}) <= minimum ({coord_desc.min()}): {result_desc[0] <= coord_desc.min()}")
print(f"- Last break ({result_desc[-1]}) >= maximum ({coord_desc.max()}): {result_desc[-1] >= coord_desc.max()}")
print()

# Test with ascending sorted coordinates (should work correctly)
coord_asc = np.array([-1.0, 0.0, 1.0])
result_asc = _infer_interval_breaks(coord_asc)

print("Test with ascending sorted coordinates (control case):")
print(f"Input coordinates: {coord_asc}")
print(f"Data range: [{coord_asc.min()}, {coord_asc.max()}]")
print(f"Interval breaks: {result_asc}")
print(f"Breaks range: [{result_asc[0]}, {result_asc[-1]}]")
print(f"- First break ({result_asc[0]}) <= minimum ({coord_asc.min()}): {result_asc[0] <= coord_asc.min()}")
print(f"- Last break ({result_asc[-1]}) >= maximum ({coord_asc.max()}): {result_asc[-1] >= coord_asc.max()}")
```

<details>

<summary>
AssertionError: Interval breaks don't cover data range for non-ascending coordinates
</summary>
```
Input coordinates: [ 0. -1.]
Data range: [-1.0, 0.0]
Interval breaks: [ 0.5 -0.5 -1.5]
Breaks range: [0.5, -1.5]

Analysis:
- First break (0.5) should be <= minimum value (-1.0)
  Result: 0.5 <= -1.0 is False
- Last break (-1.5) should be >= maximum value (0.0)
  Result: -1.5 >= 0.0 is False

Test with descending sorted coordinates:
Input coordinates: [ 1.  0. -1.]
Data range: [-1.0, 1.0]
Interval breaks: [ 1.5  0.5 -0.5 -1.5]
Breaks range: [1.5, -1.5]
- First break (1.5) <= minimum (-1.0): False
- Last break (-1.5) >= maximum (1.0): False

Test with ascending sorted coordinates (control case):
Input coordinates: [-1.  0.  1.]
Data range: [-1.0, 1.0]
Interval breaks: [-1.5 -0.5  0.5  1.5]
Breaks range: [-1.5, 1.5]
- First break (-1.5) <= minimum (-1.0): True
- Last break (1.5) >= maximum (1.0): True
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical property that interval breaks must bound the data they represent. The function incorrectly assumes that `coord[0]` is the minimum and `coord[-1]` is the maximum value (lines 888-889 in utils.py), which is only true for ascending sorted arrays.

**Specific violations of expected behavior:**

1. **Mathematical Correctness**: For interval breaks to properly represent data boundaries in visualization functions (pcolormesh, contourf), they must satisfy: `breaks[0] ≤ min(data)` and `breaks[-1] ≥ max(data)`. This invariant is violated.

2. **Silent Data Loss in Visualizations**: When these incorrect breaks are used by plotting functions, data points outside the break range are not rendered, leading to incomplete or misleading visualizations without any warning.

3. **Inconsistent Behavior with Monotonic Check**: The `_is_monotonic` function (lines 831-850) correctly identifies both ascending AND descending sequences as monotonic. However, the interval calculation only works for ascending sequences. This means `check_monotonic=True` doesn't catch descending coordinates, yet the function still produces incorrect results for them.

4. **Documentation Gap**: The docstring examples only show ascending sorted inputs, but there's no explicit requirement that coordinates must be ascending. Users reasonably expect the function to handle any coordinate order, especially since descending coordinates are common in scientific data (atmospheric pressure levels, ocean depths, etc.).

## Relevant Context

The bug is particularly problematic in scientific computing contexts where descending coordinates are standard:
- **Atmospheric data**: Pressure levels typically decrease with altitude (1000mb, 850mb, 500mb, ...)
- **Ocean data**: Depth measurements increase downward (0m, -10m, -100m, ...)
- **Time series**: Reverse chronological data

The function is used internally by public xarray plotting functions including `pcolormesh()` and `contourf()`. While `_infer_interval_breaks` is private (underscore prefix), its incorrect behavior directly impacts public API functionality.

Related GitHub issue #1852 ("2D pcolormesh plots are wrong when coordinate is not ascending order") indicates developers previously recognized similar issues but the fix appears incomplete.

Function location: `/xarray/plot/utils.py:851-899`

## Proposed Fix

The most robust fix is to ensure interval breaks are computed based on actual min/max values rather than positional assumptions:

```diff
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -885,8 +885,15 @@ def _infer_interval_breaks(coord, axis=0, scale=None, check_monotonic=False):
     deltas = 0.5 * np.diff(coord, axis=axis)
     if deltas.size == 0:
         deltas = np.array(0.0)
-    first = np.take(coord, [0], axis=axis) - np.take(deltas, [0], axis=axis)
-    last = np.take(coord, [-1], axis=axis) + np.take(deltas, [-1], axis=axis)
+
+    # Handle both ascending and descending coordinates correctly
+    coord_min = np.min(coord, axis=axis, keepdims=True)
+    coord_max = np.max(coord, axis=axis, keepdims=True)
+    delta_abs = np.abs(np.take(deltas, [0], axis=axis))
+
+    # Ensure breaks extend beyond actual data range
+    first = coord_min - delta_abs
+    last = coord_max + delta_abs
+
     trim_last = tuple(
         slice(None, -1) if n == axis else slice(None) for n in range(coord.ndim)
     )
```

Alternative approach: Make the sorting requirement explicit by defaulting `check_monotonic=True` and updating the error message to specify "ascending order" rather than just "increasing order".