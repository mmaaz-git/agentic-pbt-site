# Bug Report: dask.dataframe.tseries.resample Non-Monotonic Output Divisions

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function produces non-monotonic output divisions when `label='right'` is combined with certain division patterns, violating the fundamental requirement that divisions must be monotonically increasing.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@st.composite
def divisions_strategy(draw):
    n_divs = draw(st.integers(min_value=2, max_value=20))
    start = draw(st.datetimes(
        min_value=pd.Timestamp("2000-01-01"),
        max_value=pd.Timestamp("2020-01-01")
    ))
    freq = draw(st.sampled_from(['1h', '1D', '1min', '30min', '1W']))
    divisions = pd.date_range(start=start, periods=n_divs, freq=freq)
    return tuple(divisions)


@given(
    divisions=divisions_strategy(),
    rule=st.sampled_from(['1h', '2h', '1D', '2D', '1W', '30min', '15min']),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@settings(max_examples=1000)
def test_resample_bin_and_out_divs_monotonic(divisions, rule, closed, label):
    newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)

    for i in range(len(outdivs) - 1):
        assert outdivs[i] <= outdivs[i+1], f"outdivs not monotonic: {outdivs[i]} > {outdivs[i+1]}"
```

**Failing input**:
```python
divisions = (Timestamp('2001-02-04 00:00:00'), Timestamp('2001-02-04 01:00:00'))
rule = '1W'
closed = 'right'
label = 'right'
```

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

divisions = (pd.Timestamp('2001-02-04 00:00:00'), pd.Timestamp('2001-02-04 01:00:00'))
rule = '1W'
closed = 'right'
label = 'right'

newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)

print(f"outdivs: {outdivs}")
print(f"Monotonic: {all(outdivs[i] <= outdivs[i+1] for i in range(len(outdivs)-1))}")
```

**Output:**
```
outdivs: (Timestamp('2001-02-04 00:00:00'), Timestamp('2001-01-28 00:00:00'))
Monotonic: False
```

## Why This Is A Bug

The output divisions (`outdivs`) are `(2001-02-04, 2001-01-28)`, where the second timestamp is *before* the first one. This violates the fundamental requirement that divisions must be monotonically increasing, which is critical for dask's partitioning system. Non-monotonic divisions will cause incorrect data partitioning and potential data corruption in downstream operations.

## Fix

The bug is in the end adjustment logic. When `label='right'`, `outdivs` is shifted forward by `rule` from `tempdivs`. However, when appending the final division, the code appends `temp.index[-1]` (the unshifted value) instead of maintaining the shift.

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -104,7 +104,10 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         if outdivs[-1] > divisions[-1]:
             setter(outdivs, outdivs[-1])
         elif outdivs[-1] < divisions[-1]:
-            setter(outdivs, temp.index[-1])
+            if g.label == "right":
+                setter(outdivs, temp.index[-1] + rule)
+            else:
+                setter(outdivs, temp.index[-1])

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```

This ensures that when `label='right'`, the appended value is also shifted forward by `rule`, maintaining consistency with the earlier shift and preserving monotonicity.