# Bug Report: xarray.plot.utils._interval_to_bound_points Incorrect handling of non-contiguous intervals

**Target**: `xarray.plot.utils._interval_to_bound_points`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_interval_to_bound_points` function silently produces incorrect results when given non-contiguous intervals, violating its documented contract to return "the Intervals' boundaries".

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from xarray.plot.utils import _interval_to_bound_points


@st.composite
def non_contiguous_intervals(draw):
    n = draw(st.integers(min_value=2, max_value=10))
    intervals = []
    current_pos = 0
    for i in range(n):
        left = current_pos
        width = draw(st.floats(min_value=0.1, max_value=10))
        right = left + width
        intervals.append(pd.Interval(left, right))
        gap = draw(st.floats(min_value=0.1, max_value=5))
        current_pos = right + gap
    return intervals


@given(non_contiguous_intervals())
def test_interval_to_bound_points_completeness(intervals):
    result = _interval_to_bound_points(intervals)

    expected_bounds = []
    for interval in intervals:
        expected_bounds.append(interval.left)
    expected_bounds.append(intervals[-1].right)

    if not all(intervals[i].right == intervals[i+1].left for i in range(len(intervals)-1)):
        actual_all_bounds = []
        for interval in intervals:
            actual_all_bounds.extend([interval.left, interval.right])

        for bound in actual_all_bounds:
            assert bound in result, f"Missing boundary {bound} in result {result}"
```

**Failing input**: Any array of non-contiguous intervals, e.g., `[Interval(0, 2), Interval(5, 7)]`

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
from xarray.plot.utils import _interval_to_bound_points

intervals = [
    pd.Interval(0, 2),
    pd.Interval(5, 7),
    pd.Interval(10, 12)
]

result = _interval_to_bound_points(intervals)
print(f"Input intervals: {intervals}")
print(f"Result: {result}")
print(f"Expected: [0, 2, 5, 7, 10, 12] (all boundaries)")
print(f"Actual: [0, 5, 10, 12] (missing 2 and 7!)")

assert np.array_equal(result, [0, 2, 5, 7, 10, 12])
```

## Why This Is A Bug

The function's docstring states it "returns an array with the Intervals' boundaries" which implies it should return ALL boundaries. However, the implementation only returns the left boundaries plus the last right boundary:

```python
def _interval_to_bound_points(array: Sequence[pd.Interval]) -> np.ndarray:
    """
    Helper function which returns an array
    with the Intervals' boundaries.
    """

    array_boundaries = np.array([x.left for x in array])
    array_boundaries = np.concatenate((array_boundaries, np.array([array[-1].right])))

    return array_boundaries
```

This implementation has an **implicit assumption** that intervals are contiguous (i.e., `interval[i].right == interval[i+1].left`). However:

1. The function doesn't validate this assumption
2. The documentation doesn't mention this requirement
3. Users can easily create non-contiguous interval coordinates using pandas
4. When the assumption is violated, the function silently produces incorrect results

This is a **contract violation** because the function promises to return "the Intervals' boundaries" but fails to do so for non-contiguous intervals.

## Fix

The function should either:

**Option 1**: Validate the assumption and raise an informative error:

```diff
def _interval_to_bound_points(array: Sequence[pd.Interval]) -> np.ndarray:
    """
    Helper function which returns an array
-   with the Intervals' boundaries.
+   with the Intervals' boundaries. Assumes intervals are contiguous.
+
+   Parameters
+   ----------
+   array : Sequence[pd.Interval]
+       Sequence of contiguous intervals (interval[i].right == interval[i+1].left)
+
+   Raises
+   ------
+   ValueError
+       If intervals are not contiguous
    """

    array_boundaries = np.array([x.left for x in array])
+
+   # Validate that intervals are contiguous
+   for i in range(len(array) - 1):
+       if array[i].right != array[i+1].left:
+           raise ValueError(
+               f"Intervals must be contiguous. "
+               f"Gap found: interval[{i}].right ({array[i].right}) != "
+               f"interval[{i+1}].left ({array[i+1].left})"
+           )
+
    array_boundaries = np.concatenate((array_boundaries, np.array([array[-1].right])))

    return array_boundaries
```

**Option 2**: Return all unique boundaries (more permissive):

```diff
def _interval_to_bound_points(array: Sequence[pd.Interval]) -> np.ndarray:
    """
    Helper function which returns an array
    with the Intervals' boundaries.
    """

-   array_boundaries = np.array([x.left for x in array])
-   array_boundaries = np.concatenate((array_boundaries, np.array([array[-1].right])))
+   # Collect all boundaries (left and right)
+   boundaries = []
+   for interval in array:
+       boundaries.append(interval.left)
+       boundaries.append(interval.right)
+
+   # Remove duplicates and sort
+   array_boundaries = np.unique(boundaries)

    return array_boundaries
```

Option 1 is recommended as it maintains backward compatibility for valid inputs while failing fast on invalid inputs.