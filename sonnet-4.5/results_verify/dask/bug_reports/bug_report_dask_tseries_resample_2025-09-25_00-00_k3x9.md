# Bug Report: dask.dataframe.tseries Unsorted Output Divisions

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function returns unsorted `outdivs` when resampling with `label='right'` in certain conditions, violating the invariant that divisions must be sorted.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@st.composite
def date_ranges(draw):
    start_year = draw(st.integers(min_value=2000, max_value=2023))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))
    periods = draw(st.integers(min_value=2, max_value=100))
    freq = draw(st.sampled_from(['h', 'D', '30min', '2h', '6h']))
    start = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    dates = pd.date_range(start, periods=periods, freq=freq)
    return dates


@st.composite
def resample_rules(draw):
    return draw(st.sampled_from(['30min', 'h', '2h', 'D', 'W']))


@given(
    dates=date_ranges(),
    rule=resample_rules(),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@settings(max_examples=500)
def test_resample_bin_and_out_divs_returns_sorted_divisions(dates, rule, closed, label):
    divisions = list(dates)
    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)
    except Exception:
        assume(False)

    for i in range(len(outdivs) - 1):
        assert outdivs[i] <= outdivs[i + 1], f"outdivs not sorted: {outdivs[i]} > {outdivs[i + 1]}"
```

**Failing input**:
- `dates`: DatetimeIndex from 2001-02-03 00:00:00 with 26 hourly periods
- `rule`: 'W'
- `closed`: 'right'
- `label`: 'right'

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

dates = pd.date_range('2001-02-03 00:00:00', periods=26, freq='h')
divisions = list(dates)

newdivs, outdivs = _resample_bin_and_out_divs(divisions, 'W', closed='right', label='right')

print("outdivs:", outdivs)
assert outdivs[0] <= outdivs[1]
```

**Output:**
```
outdivs: (Timestamp('2001-02-04 00:00:00'), Timestamp('2001-01-28 00:00:00'))
AssertionError: Feb 4 is not <= Jan 28
```

## Why This Is A Bug

Divisions in Dask DataFrames must always be sorted to maintain the fundamental invariant that partitions are ordered. This function is responsible for calculating divisions for resampled time series, and returning unsorted divisions will cause downstream operations to fail or produce incorrect results.

The bug occurs when:
1. `label='right'` (so `outdivs = tempdivs + rule`, shifting timestamps forward)
2. The adjustment logic at the end uses `append` mode
3. It appends `temp.index[-1]` (an earlier timestamp) after `outdivs[0]` (a later timestamp)

## Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -95,10 +95,15 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         else:
             setter = lambda a, val: a.__setitem__(-1, val)
         setter(newdivs, divisions[-1] + res)
+
         if outdivs[-1] > divisions[-1]:
             setter(outdivs, outdivs[-1])
         elif outdivs[-1] < divisions[-1]:
-            setter(outdivs, temp.index[-1])
+            # When appending, we need to append the correct value
+            # temp.index[-1] is the last resampling bin start, which should be
+            # placed correctly based on the label
+            final_outdiv = outdivs[-1] if len(newdivs) >= len(divs) else temp.index[-1]
+            setter(outdivs, final_outdiv)

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```

**Alternative simpler fix:** Ensure outdivs is sorted before returning:

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -100,6 +100,11 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         elif outdivs[-1] < divisions[-1]:
             setter(outdivs, temp.index[-1])

+    # Ensure divisions are sorted
+    newdivs = sorted(newdivs)
+    outdivs = sorted(outdivs)
+
     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```