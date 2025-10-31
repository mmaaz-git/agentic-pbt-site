# Bug Report: xarray.plot.utils._infer_interval_breaks Incorrect Results for Non-Monotonic Data

**Target**: `xarray.plot.utils._infer_interval_breaks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_infer_interval_breaks` function produces mathematically incorrect interval breaks for non-monotonic coordinate data, resulting in breaks that don't properly contain all the original coordinate values.

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

# Run the test
test_infer_interval_breaks_contains_original()
```

<details>

<summary>
**Failing input**: `[0.0, 1.0, 0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 21, in <module>
    test_infer_interval_breaks_contains_original()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 6, in test_infer_interval_breaks_contains_original
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 17, in test_infer_interval_breaks_contains_original
    assert min_break <= arr[i] <= max_break, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Original value 1.0 not in range [-0.5, 0.5]
Falsifying example: test_infer_interval_breaks_contains_original(
    values=[0.0, 1.0, 0.0],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import xarray.plot.utils as plot_utils

coord = np.array([0.0, 1.0, 0.0])
breaks = plot_utils._infer_interval_breaks(coord)

print(f"Input coordinates: {coord}")
print(f"Inferred breaks:   {breaks}")
print(f"Breaks range: [{min(breaks)}, {max(breaks)}]")

# Check if all coordinates are contained in the breaks range
for c in coord:
    if not (min(breaks) <= c <= max(breaks)):
        print(f"ERROR: coordinate value {c} not in range [{min(breaks)}, {max(breaks)}]")

# Also show with check_monotonic=True
print("\nWith check_monotonic=True:")
try:
    breaks_checked = plot_utils._infer_interval_breaks(coord, check_monotonic=True)
    print(f"Breaks: {breaks_checked}")
except ValueError as e:
    print(f"Error raised: {e}")
```

<details>

<summary>
Output showing coordinate value not in breaks range
</summary>
```
Input coordinates: [0. 1. 0.]
Inferred breaks:   [-0.5  0.5  0.5 -0.5]
Breaks range: [-0.5, 0.5]
ERROR: coordinate value 1.0 not in range [-0.5, 0.5]

With check_monotonic=True:
Error raised: The input coordinate is not sorted in increasing order along axis 0. This can lead to unexpected results. Consider calling the `sortby` method on the input DataArray. To plot data with categorical axes, consider using the `heatmap` function from the `seaborn` statistical plotting library.
```
</details>

## Why This Is A Bug

The `_infer_interval_breaks` function computes interval breaks using local differences (`0.5 * np.diff(coord)`). For non-monotonic data like `[0.0, 1.0, 0.0]`:

1. `np.diff([0.0, 1.0, 0.0])` produces `[1.0, -1.0]`
2. `deltas = 0.5 * [1.0, -1.0] = [0.5, -0.5]`
3. The breaks are constructed as:
   - `first = coord[0] - deltas[0] = 0.0 - 0.5 = -0.5`
   - `middle = coord[:-1] + deltas = [0.0, 1.0] + [0.5, -0.5] = [0.5, 0.5]`
   - `last = coord[-1] + deltas[-1] = 0.0 + (-0.5) = -0.5`
   - Result: `[-0.5, 0.5, 0.5, -0.5]`

The resulting breaks range `[-0.5, 0.5]` fails to contain the middle coordinate value `1.0`. This violates the fundamental property that interval breaks should encompass all coordinate values for proper visualization in plotting functions like `pcolormesh`.

While the function has a `check_monotonic` parameter that can detect and reject non-monotonic data, it defaults to `False`, allowing users to receive incorrect results without any warning. The documentation only shows monotonic examples and doesn't explicitly state that monotonic input is required for correct results.

## Relevant Context

The function is located in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/plot/utils.py:853`.

When used properly through xarray's plotting functions like `pcolormesh`, the function is called with `check_monotonic=True` (see line 2306 and 2314 in `dataarray_plot.py`), which prevents the issue. However, the function can be imported and used directly, and its default behavior produces incorrect results.

The underscore prefix indicates this is an internal function, but it remains accessible for direct import. The algorithm inherently assumes monotonic data based on its mathematical design using consecutive differences.

## Proposed Fix

The simplest fix is to change the default parameter to enforce monotonic checking:

```diff
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -853,7 +853,7 @@ def _is_monotonic(coord, axis=0):
         return np.all(delta_pos) or np.all(delta_neg)


-def _infer_interval_breaks(coord, axis=0, scale=None, check_monotonic=False):
+def _infer_interval_breaks(coord, axis=0, scale=None, check_monotonic=True):
     """
     >>> _infer_interval_breaks(np.arange(5))
     array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])
```

This ensures the function raises a clear error when given non-monotonic data, preventing silent incorrect results. Alternatively, the function could be enhanced to handle non-monotonic data correctly, or the documentation could clearly state the monotonic requirement.