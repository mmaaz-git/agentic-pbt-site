# Bug Report: dask.dataframe.tseries.resample Duplicate Timestamps Corrupt Division Boundaries

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function produces duplicate timestamps in output divisions when resampling with certain parameter combinations, violating Dask's requirement for monotonically increasing unique divisions and causing downstream operations to fail.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, assume, settings
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@given(
    st.integers(min_value=3, max_value=100),
    st.sampled_from(['h', 'D', '2h', '3h', '6h', '12h', '30min', 'W', '2D', '3D', '5D', '7D', '10D']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right'])
)
@settings(max_examples=5000, deadline=None)
def test_outdivs_no_consecutive_duplicates(n_divisions, rule, closed, label):
    start = pd.Timestamp('2020-01-01')
    end = pd.Timestamp('2021-12-31')
    divisions = pd.date_range(start, end, periods=n_divisions)

    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)
    except Exception:
        assume(False)

    for i in range(len(outdivs) - 1):
        if outdivs[i] == outdivs[i + 1]:
            assert False, f"Found consecutive duplicate in outdivs at index {i}: {outdivs[i]}"


if __name__ == "__main__":
    # Run the property test
    test_outdivs_no_consecutive_duplicates()
```

<details>

<summary>
**Failing input**: `n_divisions=75, rule='10D', closed='right', label='left'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 30, in <module>
    test_outdivs_no_consecutive_duplicates()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 7, in test_outdivs_no_consecutive_duplicates
    st.integers(min_value=3, max_value=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 25, in test_outdivs_no_consecutive_duplicates
    assert False, f"Found consecutive duplicate in outdivs at index {i}: {outdivs[i]}"
           ^^^^^
AssertionError: Found consecutive duplicate in outdivs at index 73: 2021-12-21 00:00:00
Falsifying example: test_outdivs_no_consecutive_duplicates(
    n_divisions=75,
    rule='10D',
    closed='right',
    label='left',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/59/hypo.py:25
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

start = pd.Timestamp('2020-01-01')
end = pd.Timestamp('2021-12-31')
divisions = pd.date_range(start, end, periods=88)

newdivs, outdivs = _resample_bin_and_out_divs(divisions, '10D', closed='right', label='left')

print(f"Number of divisions: {len(divisions)}")
print(f"Number of newdivs: {len(newdivs)}")
print(f"Number of outdivs: {len(outdivs)}")
print(f"\noutdivs[-5:] = {outdivs[-5:]}")

# Check for duplicates
for i in range(len(outdivs) - 1):
    if outdivs[i] == outdivs[i + 1]:
        print(f"\nDUPLICATE FOUND at index {i}: {outdivs[i]}")
        print(f"  outdivs[{i}] = {outdivs[i]}")
        print(f"  outdivs[{i+1}] = {outdivs[i+1]}")

# Show the last few elements to understand the problem
print(f"\ndivisions[-1] = {divisions[-1]}")
print(f"newdivs[-3:] = {newdivs[-3:]}")
print(f"outdivs[-3:] = {outdivs[-3:]}")
```

<details>

<summary>
Duplicate timestamp found in output divisions
</summary>
```
Number of divisions: 88
Number of newdivs: 75
Number of outdivs: 75

outdivs[-5:] = (Timestamp('2021-11-21 00:00:00'), Timestamp('2021-12-01 00:00:00'), Timestamp('2021-12-11 00:00:00'), Timestamp('2021-12-21 00:00:00'), Timestamp('2021-12-21 00:00:00'))

DUPLICATE FOUND at index 73: 2021-12-21 00:00:00
  outdivs[73] = 2021-12-21 00:00:00
  outdivs[74] = 2021-12-21 00:00:00

divisions[-1] = 2021-12-31 00:00:00
newdivs[-3:] = (Timestamp('2021-12-11 00:00:00.000000001'), Timestamp('2021-12-21 00:00:00.000000001'), Timestamp('2021-12-31 00:00:00.000000001'))
outdivs[-3:] = (Timestamp('2021-12-11 00:00:00'), Timestamp('2021-12-21 00:00:00'), Timestamp('2021-12-21 00:00:00'))
```
</details>

## Why This Is A Bug

This bug violates Dask's fundamental requirement that divisions be "in ascending order" (from `dask.dataframe.DataFrame.divisions` documentation), which implies unique values. The duplicate timestamps in `outdivs` break this invariant in multiple ways:

1. **Documentation Violation**: Dask explicitly documents that divisions must be "Tuple of npartitions + 1 values, in ascending order". Duplicate values violate ascending order by definition.

2. **Functional Impact**: The duplicate divisions cause actual failures in resample operations. When attempting to resample a Dask DataFrame with these parameters, the operation fails with: `"Index is not contained within new index"` - a misleading error that doesn't indicate the real problem.

3. **Logic Error**: The bug occurs in the "Adjust ends" section (lines 89-102 of resample.py). When `len(newdivs) < len(divs)` (line 93), the code uses `append` to add elements. In the problematic case where `outdivs[-1] < divisions[-1]` (line 100), it appends `temp.index[-1]` (line 101). However, `temp.index[-1]` equals the current `outdivs[-1]` value (both are `2021-12-21 00:00:00`), creating a duplicate.

4. **Downstream Effects**: Duplicate divisions break partition boundaries, potentially causing:
   - Incorrect results in loc operations
   - Failures in merge and groupby operations
   - Data corruption in time series analysis

## Relevant Context

The bug manifests when all these conditions align:
- The resample rule creates fewer output divisions than input divisions (`len(newdivs) < len(divs)`)
- Using `closed='right'` with `label='left'` parameters
- The last resampled bin boundary falls before the last input division

This commonly occurs with multi-day resampling periods (e.g., '10D') on datasets spanning 1-2 years with many partitions.

The Dask resample implementation relies on pandas' `Grouper` and `resample` functionality to determine bin boundaries. The adjustment logic at the end tries to ensure divisions cover the full data range but incorrectly handles the append case.

## Proposed Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -98,7 +98,8 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         if outdivs[-1] > divisions[-1]:
             setter(outdivs, outdivs[-1])
         elif outdivs[-1] < divisions[-1]:
-            setter(outdivs, temp.index[-1])
+            if len(newdivs) >= len(divs) or outdivs[-1] != temp.index[-1]:
+                setter(outdivs, temp.index[-1])

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```