# Bug Report: pandas.core.arrays IntervalArray.unique() Drops Intervals with Negative Breaks

**Target**: `pandas.core.arrays.IntervalArray.unique()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`IntervalArray.unique()` incorrectly drops distinct intervals when all break values are negative. For example, an array with intervals `[-3, -2)` and `[-2, -1)` returns only `[-3, -2)` after calling `unique()`, silently losing the second interval.

## Property-Based Test

```python
import pandas as pd
from pandas.core.arrays import IntervalArray
from hypothesis import given, strategies as st, assume, settings


@settings(max_examples=500)
@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=30),
    st.sampled_from(['left', 'right', 'both', 'neither'])
)
def test_intervalarray_unique_preserves_distinct_intervals(breaks, closed):
    breaks_sorted = sorted(set(breaks))
    assume(len(breaks_sorted) >= 2)

    arr = IntervalArray.from_breaks(breaks_sorted, closed=closed)
    combined = IntervalArray._concat_same_type([arr, arr])
    unique_arr = combined.unique()

    manual_unique = set()
    for interval in combined:
        manual_unique.add((interval.left, interval.right, interval.closed))

    assert len(unique_arr) == len(manual_unique), \
        f"unique() returned {len(unique_arr)} intervals but there are {len(manual_unique)} distinct intervals"
```

**Failing input**: `breaks=[-3, -2, -1], closed='left'`

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.arrays import IntervalArray

arr = IntervalArray.from_breaks([-3, -2, -1], closed='left')
print(f"Original intervals: {list(arr)}")

combined = IntervalArray._concat_same_type([arr, arr])
print(f"Concatenated: {list(combined)}")

unique = combined.unique()
print(f"After unique(): {list(unique)}")
print(f"Length: {len(unique)} (expected 2)")
```

Output:
```
Original intervals: [Interval(-3, -2, closed='left'), Interval(-2, -1, closed='left')]
Concatenated: [Interval(-3, -2, closed='left'), Interval(-2, -1, closed='left'), Interval(-3, -2, closed='left'), Interval(-2, -1, closed='left')]
After unique(): [Interval(-3, -2, closed='left')]
Length: 1 (expected 2)
```

## Why This Is A Bug

This is a serious data loss bug:

1. **Silent data corruption**: `unique()` silently drops distinct intervals with no warning or error, potentially causing incorrect analysis results.

2. **Inconsistent behavior**: The bug only occurs when all break values are negative. Positive breaks work correctly:
   - `[-3, -2, -1]`: ✗ Returns 1 interval instead of 2
   - `[1, 2, 3]`: ✓ Correctly returns 2 intervals
   - `[-2, -1, 0]`: ✓ Correctly returns 2 intervals
   - `[-10, -9, -8]`: ✗ Returns 1 interval instead of 2

3. **Violates fundamental uniqueness property**: For any array, `unique(concat([arr, arr]))` should equal `arr`, but this fails for intervals with all-negative breaks.

4. **Real-world impact**: Users working with data containing negative intervals (e.g., temperature ranges, financial data, coordinate systems) will experience silent data loss when deduplicating intervals.

## Fix

The bug is likely in the comparison or hashing logic used by `unique()` for IntervalArray when dealing with negative values. The issue may be in how intervals are compared or how the unique operation identifies duplicates.

Investigation needed in `pandas/core/arrays/interval.py`:

```python
def unique(self) -> Self:
    # Current implementation appears to have issues with negative intervals
    # Likely in the factorize or unique value detection logic
```

Without access to the exact implementation, the fix would involve:
1. Ensuring interval comparison works correctly for negative values
2. Verifying the hashing/equality logic in `_interval_shared.pyx` or wherever intervals are compared
3. Adding test cases for negative break values in the test suite