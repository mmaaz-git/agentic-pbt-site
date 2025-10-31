# Bug Report: dask.dataframe.tseries Duplicate Divisions

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function can produce output divisions (`outdivs`) with duplicate consecutive values, violating the dask dataframe invariant that divisions must be strictly monotonically increasing.

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
def test_resample_divisions_monotonic(n_divs, freq, closed, label):
    start = pd.Timestamp('2000-01-01')
    end = start + pd.Timedelta(days=30)
    divisions = pd.date_range(start, end, periods=n_divs)

    newdivs, outdivs = _resample_bin_and_out_divs(divisions, freq, closed=closed, label=label)

    for i in range(len(outdivs) - 1):
        assert outdivs[i] < outdivs[i+1], f"outdivs not monotonic: {outdivs[i]} >= {outdivs[i+1]}"
```

**Failing input**: `n_divs=7, freq='W', closed='left', label='left'`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

start = pd.Timestamp('2000-01-01')
end = start + pd.Timedelta(days=30)
divisions = pd.date_range(start, end, periods=7)

newdivs, outdivs = _resample_bin_and_out_divs(divisions, 'W', closed='left', label='left')

print("outdivs:", outdivs)

for i in range(len(outdivs) - 1):
    if outdivs[i] >= outdivs[i+1]:
        print(f"ERROR: outdivs[{i}] = {outdivs[i]} >= outdivs[{i+1}] = {outdivs[i+1]}")
```

## Why This Is A Bug

Dask dataframes require divisions to be strictly monotonically increasing. When `outdivs` contains duplicate consecutive values, it violates this fundamental invariant. This can cause:

1. Incorrect partition boundaries in the resampled dataframe
2. Errors in downstream operations that assume strictly increasing divisions
3. Data corruption or incorrect results when slicing by time ranges

The bug occurs in lines 98-99 of `resample.py`:

```python
if outdivs[-1] > divisions[-1]:
    setter(outdivs, outdivs[-1])
```

When `setter` is the `append` lambda (line 94), this appends `outdivs[-1]` to the list, creating a duplicate of the already-existing last element.

## Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -95,9 +95,7 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         else:
             setter = lambda a, val: a.__setitem__(-1, val)
         setter(newdivs, divisions[-1] + res)
-        if outdivs[-1] > divisions[-1]:
-            setter(outdivs, outdivs[-1])
-        elif outdivs[-1] < divisions[-1]:
+        if outdivs[-1] < divisions[-1]:
             setter(outdivs, temp.index[-1])

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```

The fix removes the problematic branch that attempts to append/set `outdivs[-1]` when it's already greater than `divisions[-1]`. In this case, `outdivs[-1]` is already correct and doesn't need adjustment.