# Bug Report: dask.dataframe.tseries Resample Division Duplicates

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function produces duplicate consecutive timestamps in output divisions when the resampling frequency is larger than the input time range, violating the strict monotonicity requirement for Dask divisions.

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
    rule=st.sampled_from(['1h', '2h', '6h', '12h', '1D', '2D']),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@settings(max_examples=500, deadline=None)
def test_no_duplicate_divisions(divisions, rule, closed, label):
    newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)

    for i in range(len(outdivs) - 1):
        assert outdivs[i] < outdivs[i+1], \
            f"Divisions not strictly increasing: outdivs[{i}]={outdivs[i]}, outdivs[{i+1}]={outdivs[i+1]}"
```

**Failing input**:
```
divisions=[Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-01 01:00:00')]
rule='2h'
closed='left'
label='left'
```

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

divisions = [pd.Timestamp('2000-01-01 00:00:00'), pd.Timestamp('2000-01-01 01:00:00')]
newdivs, outdivs = _resample_bin_and_out_divs(divisions, '2h', closed='left', label='left')

print(f"outdivs: {outdivs}")
for i in range(len(outdivs) - 1):
    if outdivs[i] == outdivs[i+1]:
        print(f"DUPLICATE: outdivs[{i}] == outdivs[{i+1}] == {outdivs[i]}")
```

**Output:**
```
outdivs: (Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-01 00:00:00'))
DUPLICATE: outdivs[0] == outdivs[1] == 2000-01-01 00:00:00
```

## Why This Is A Bug

Dask requires divisions to be strictly monotonically increasing to properly partition data. Duplicate consecutive timestamps create zero-width partitions, which:
1. Violate Dask's fundamental partitioning assumptions
2. Lead to ambiguous partition assignment for data points
3. May cause downstream operations to fail or produce incorrect results

This bug occurs when the input time range is smaller than the resampling frequency (e.g., 1 hour of data resampled to 2-hour bins). The pandas resampling in the function creates a single bin containing all input data, but the end-adjustment logic (lines 89-101) incorrectly extends the divisions without ensuring strict monotonicity.

## Fix

The root cause is that the end-adjustment logic needs to ensure strict monotonicity when modifying division boundaries. When the resampling frequency exceeds the input range, the function should either:

1. Ensure outdivs remains strictly monotonic by checking before any adjustment
2. Return a minimal valid division set (e.g., just two distinct timestamps)

A potential fix:

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -97,8 +97,11 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         setter(newdivs, divisions[-1] + res)
         if outdivs[-1] > divisions[-1]:
             setter(outdivs, outdivs[-1])
         elif outdivs[-1] < divisions[-1]:
-            setter(outdivs, temp.index[-1])
+            new_val = temp.index[-1]
+            # Ensure strict monotonicity
+            if len(outdivs) > 1 and new_val <= outdivs[-2]:
+                new_val = outdivs[-2] + rule
+            setter(outdivs, new_val)

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```

However, this fix may still not be complete as the underlying issue is that the function doesn't properly handle edge cases where the resampling frequency is comparable to or larger than the input time range. A more comprehensive review of the division computation logic is needed.