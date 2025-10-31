# Bug Report: dask.dataframe.tseries Output Divisions Extend Before Input Boundaries

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function produces output divisions (`outdivs`) where the first element can be before the first input division, incorrectly extending the time range beyond the input data boundaries.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@given(
    st.integers(min_value=2, max_value=20),
    st.sampled_from(['h', 'D', '2h', '3D', '12h', 'W']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right']),
)
@settings(max_examples=500)
def test_resample_divisions_contain_original_boundaries(n_divs, freq, closed, label):
    start = pd.Timestamp('2000-01-01')
    end = start + pd.Timedelta(days=30)
    divisions = pd.date_range(start, end, periods=n_divs)

    newdivs, outdivs = _resample_bin_and_out_divs(divisions, freq, closed=closed, label=label)

    assert outdivs[0] >= divisions[0], f"First outdiv {outdivs[0]} before first division {divisions[0]}"


if __name__ == "__main__":
    test_resample_divisions_contain_original_boundaries()
```

<details>

<summary>
**Failing input**: `n_divs=2, freq='h', closed='right', label='left'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 24, in <module>
    test_resample_divisions_contain_original_boundaries()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 7, in test_resample_divisions_contain_original_boundaries
    st.integers(min_value=2, max_value=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 20, in test_resample_divisions_contain_original_boundaries
    assert outdivs[0] >= divisions[0], f"First outdiv {outdivs[0]} before first division {divisions[0]}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: First outdiv 1999-12-31 23:00:00 before first division 2000-01-01 00:00:00
Falsifying example: test_resample_divisions_contain_original_boundaries(
    n_divs=2,
    freq='h',
    closed='right',
    label='left',
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Create divisions with 2 points
start = pd.Timestamp('2000-01-01')
end = start + pd.Timedelta(days=30)
divisions = pd.date_range(start, end, periods=2)

# Call the function with the failing parameters
newdivs, outdivs = _resample_bin_and_out_divs(divisions, 'h', closed='right', label='left')

print(f"Input divisions[0]: {divisions[0]}")
print(f"Input divisions[-1]: {divisions[-1]}")
print()
print(f"Output newdivs[0]: {newdivs[0]}")
print(f"Output outdivs[0]: {outdivs[0]}")
print()

# Check if outdivs extends before the input range
if outdivs[0] < divisions[0]:
    print(f"ERROR: outdivs[0] is before divisions[0]")
    print(f"  outdivs[0] = {outdivs[0]}")
    print(f"  divisions[0] = {divisions[0]}")
    print(f"  Difference: {divisions[0] - outdivs[0]}")
else:
    print("OK: outdivs[0] is within the input range")
```

<details>

<summary>
ERROR: outdivs extends 1 hour before input divisions
</summary>
```
Input divisions[0]: 2000-01-01 00:00:00
Input divisions[-1]: 2000-01-31 00:00:00

Output newdivs[0]: 2000-01-01 00:00:00
Output outdivs[0]: 1999-12-31 23:00:00

ERROR: outdivs[0] is before divisions[0]
  outdivs[0] = 1999-12-31 23:00:00
  divisions[0] = 2000-01-01 00:00:00
  Difference: 0 days 01:00:00
```
</details>

## Why This Is A Bug

This violates the expected behavior that output divisions should represent the actual boundaries of the resampled data. In Dask, divisions are documented to represent the minimum and maximum index values of each partition - they define the actual data boundaries. When `outdivs[0]` is set to a timestamp before `divisions[0]`, it incorrectly claims that the resampled data extends before the original input data range.

The bug occurs specifically when `closed='right'` and `label='left'` are used together. In the code (line 78 of `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py`), when `closed='right'`, the function adds a `res` offset to `tempdivs`, which shifts the divisions backwards by one nanosecond (for tick-based frequencies like hourly). This causes `outdivs` to start before the actual data begins.

The function already contains logic to prevent `newdivs[0]` from extending before `divisions[0]` (lines 90-91), but fails to apply the same constraint to `outdivs[0]`. This inconsistency shows that maintaining input boundaries was the intended behavior but was only partially implemented.

## Relevant Context

The `_resample_bin_and_out_divs` function is used internally by Dask's resample operations to determine how to partition resampled time series data. While this is a private function (indicated by the underscore prefix), it affects the public `resample()` API that users directly interact with.

Key code locations:
- Function definition: `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py:66-103`
- Existing boundary check for `newdivs`: lines 90-91
- Missing boundary check for `outdivs`: no corresponding check exists

The bug can cause:
1. Incorrect results when querying resampled data by time ranges
2. Confusion about the actual time span of resampled data
3. Potential index errors when operations expect divisions to accurately represent data boundaries

## Proposed Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -89,6 +89,8 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
     # Adjust ends
     if newdivs[0] < divisions[0]:
         newdivs[0] = divisions[0]
+    if outdivs[0] < divisions[0]:
+        outdivs[0] = divisions[0]
     if newdivs[-1] < divisions[-1]:
         if len(newdivs) < len(divs):
             setter = lambda a, val: a.append(val)
```