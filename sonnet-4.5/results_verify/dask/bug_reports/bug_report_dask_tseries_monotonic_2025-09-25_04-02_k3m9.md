# Bug Report: dask.dataframe.tseries Resample Division Monotonicity Violation

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function produces non-monotonic output divisions when using right-closed and right-labeled resampling with certain time ranges and frequencies, violating a fundamental invariant of time series operations.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

@st.composite
def timestamp_list_strategy(draw):
    size = draw(st.integers(min_value=2, max_value=20))
    start = draw(st.datetimes(
        min_value=pd.Timestamp('2000-01-01'),
        max_value=pd.Timestamp('2020-01-01')
    ))
    freq_hours = draw(st.integers(min_value=1, max_value=24*7))
    timestamps = pd.date_range(start=start, periods=size, freq=f'{freq_hours}h')
    return timestamps.tolist()

@given(
    divisions=timestamp_list_strategy(),
    rule=st.sampled_from(['1h', '2h', '6h', '12h', '1D', '2D', '1W']),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@settings(max_examples=500, deadline=None)
def test_resample_bin_and_out_divs_monotonic(divisions, rule, closed, label):
    newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)

    for i in range(len(outdivs) - 1):
        assert outdivs[i] <= outdivs[i+1], \
            f"outdivs not monotonic: outdivs[{i}]={outdivs[i]} > outdivs[{i+1}]={outdivs[i+1]}"
```

**Failing input**:
```
divisions=[Timestamp('2000-12-17 00:00:00'), Timestamp('2000-12-17 01:00:00')]
rule='1W'
closed='right'
label='right'
```

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

divisions = [pd.Timestamp('2000-12-17 00:00:00'), pd.Timestamp('2000-12-17 01:00:00')]
newdivs, outdivs = _resample_bin_and_out_divs(divisions, '1W', closed='right', label='right')

print(f"outdivs: {outdivs}")
print(f"outdivs[0] = {outdivs[0]}")
print(f"outdivs[1] = {outdivs[1]}")
print(f"Is monotonic? {outdivs[0] <= outdivs[1]}")
```

**Output:**
```
outdivs: (Timestamp('2000-12-17 00:00:00'), Timestamp('2000-12-10 00:00:00'))
outdivs[0] = 2000-12-17 00:00:00
outdivs[1] = 2000-12-10 00:00:00
Is monotonic? False
```

## Why This Is A Bug

Division arrays in Dask must be monotonically increasing as they represent partition boundaries for distributed time series data. Non-monotonic divisions will cause:
1. Incorrect data partitioning
2. Sorting errors in downstream operations
3. Invalid time range queries

The bug occurs in the end-adjustment logic (lines 89-101 of resample.py). When `label='right'`, line 82 sets `outdivs = tempdivs + rule`, adding the frequency offset. Later, when `outdivs[-1] < divisions[-1]`, line 101 unconditionally sets `outdivs[-1] = temp.index[-1]`, which can be much earlier than `outdivs[-2]`, breaking monotonicity.

## Fix

The fix should ensure that any adjustment to `outdivs[-1]` maintains monotonicity. One approach is to use `max()` to ensure the last element is at least as large as the second-to-last:

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -98,7 +98,7 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         if outdivs[-1] > divisions[-1]:
             setter(outdivs, outdivs[-1])
         elif outdivs[-1] < divisions[-1]:
-            setter(outdivs, temp.index[-1])
+            setter(outdivs, max(temp.index[-1], outdivs[-2] if len(outdivs) > 1 else temp.index[-1]))

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```

However, this may not be the complete fix as it could mask underlying issues with the division calculation logic. A more thorough review of the end-adjustment algorithm is recommended to ensure correctness across all parameter combinations.