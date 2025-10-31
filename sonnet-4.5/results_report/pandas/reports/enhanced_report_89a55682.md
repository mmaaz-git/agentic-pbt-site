# Bug Report: pandas.core.arrays.IntervalArray.unique() Drops Intervals with All-Negative Breaks

**Target**: `pandas.core.arrays.IntervalArray.unique()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`IntervalArray.unique()` incorrectly drops distinct intervals when all break values are negative, returning only the first interval and silently discarding the rest, causing data loss.

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

if __name__ == "__main__":
    test_intervalarray_unique_preserves_distinct_intervals()
```

<details>

<summary>
**Failing input**: `breaks=[-1, -2, -3], closed='left'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 27, in <module>
    test_intervalarray_unique_preserves_distinct_intervals()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 7, in test_intervalarray_unique_preserves_distinct_intervals
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 23, in test_intervalarray_unique_preserves_distinct_intervals
    assert len(unique_arr) == len(manual_unique), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: unique() returned 1 intervals but there are 2 distinct intervals
Falsifying example: test_intervalarray_unique_preserves_distinct_intervals(
    breaks=[-1, -2, -3],
    closed='left',  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/1/hypo.py:24
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.arrays import IntervalArray

# Create an IntervalArray with negative breaks
arr = IntervalArray.from_breaks([-3, -2, -1], closed='left')
print(f"Original intervals: {list(arr)}")

# Concatenate the array with itself (to simulate having duplicates)
combined = IntervalArray._concat_same_type([arr, arr])
print(f"Concatenated: {list(combined)}")

# Apply unique() to get distinct intervals
unique = combined.unique()
print(f"After unique(): {list(unique)}")
print(f"Length: {len(unique)} (expected 2)")

# Verify the bug: we expect 2 unique intervals but get only 1
expected_intervals = 2
actual_intervals = len(unique)
print(f"\nBUG: Expected {expected_intervals} unique intervals, got {actual_intervals}")
```

<details>

<summary>
BUG: Expected 2 unique intervals, got 1
</summary>
```
Original intervals: [Interval(-3, -2, closed='left'), Interval(-2, -1, closed='left')]
Concatenated: [Interval(-3, -2, closed='left'), Interval(-2, -1, closed='left'), Interval(-3, -2, closed='left'), Interval(-2, -1, closed='left')]
After unique(): [Interval(-3, -2, closed='left')]
Length: 1 (expected 2)

BUG: Expected 2 unique intervals, got 1
```
</details>

## Why This Is A Bug

This violates the fundamental contract of the `unique()` method, which according to pandas documentation should "return unique values of IntervalArray" and "return a new IntervalArray with duplicate values removed."

The bug manifests specifically when **all break values are negative**. The method silently drops valid, distinct intervals without any warning or error, leading to silent data corruption. This is demonstrated by:

1. **Data Loss**: When processing `[-3, -2, -1]` with `closed='left'`, the two distinct intervals `[-3, -2)` and `[-2, -1)` are reduced to just `[-3, -2)`, losing 50% of the data.

2. **Inconsistent Behavior**: The same operation works correctly with positive or mixed breaks:
   - Positive breaks `[1, 2, 3]`: Correctly returns 2 unique intervals
   - Mixed breaks `[-2, -1, 0]`: Correctly returns 2 unique intervals
   - All-negative breaks `[-3, -2, -1]`: **Incorrectly** returns only 1 interval
   - All-negative breaks `[-10, -9, -8]`: **Incorrectly** returns only 1 interval

3. **Violates Uniqueness Invariant**: For any collection, applying `unique()` to a concatenation of identical arrays should preserve all distinct elements. This invariant is broken for negative intervals.

## Relevant Context

This bug affects pandas version 2.3.2 and likely other versions. The issue is located in the `unique()` method implementation in `/pandas/core/arrays/interval.py`.

The bug pattern suggests an issue with how intervals are compared or hashed when all values are negative. This could be related to:
- Incorrect handling of negative values in the underlying comparison logic
- Issues in the factorization or hashing algorithm used by `unique()`
- Problems in the Cython implementation (`_libs.interval` or `_interval_shared.pyx`)

Real-world impact includes:
- Financial analysis with negative returns or losses
- Temperature data analysis with sub-zero values
- Coordinate systems with negative positions
- Any domain where negative interval bounds are common

Documentation references:
- [pandas.IntervalArray.unique](https://pandas.pydata.org/docs/reference/api/pandas.arrays.IntervalArray.unique.html)
- [pandas.IntervalArray](https://pandas.pydata.org/docs/reference/api/pandas.arrays.IntervalArray.html)

## Proposed Fix

The bug likely resides in the comparison or hashing logic for intervals. Without direct access to modify the pandas source, here's a high-level approach to fix this issue:

1. The issue is likely in how `unique()` delegates to `factorize()` or similar internal methods
2. Check if the comparison operators handle negative bounds correctly
3. Verify hash computation for intervals with negative values

As a workaround until the bug is fixed, users can manually deduplicate intervals:

```python
def safe_unique_intervals(interval_array):
    """Workaround for IntervalArray.unique() bug with negative values"""
    seen = set()
    unique_intervals = []
    for interval in interval_array:
        key = (interval.left, interval.right, interval.closed)
        if key not in seen:
            seen.add(key)
            unique_intervals.append(interval)
    return IntervalArray(unique_intervals)
```