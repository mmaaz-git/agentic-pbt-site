# Bug Report: dask.dataframe.tseries.resample Unsorted Output Divisions

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function returns unsorted `outdivs` when `label='right'` and certain time boundary conditions are met, violating the fundamental invariant that divisions must be sorted.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings, assume

from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@st.composite
def timestamp_divisions(draw):
    n = draw(st.integers(min_value=2, max_value=20))
    start = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2024, 1, 1)
    ))
    freq_value = draw(st.integers(min_value=1, max_value=100))
    freq_unit = draw(st.sampled_from(['min', 'h', 'D']))
    divisions = pd.date_range(start=start, periods=n, freq=f'{freq_value}{freq_unit}')
    return divisions


@st.composite
def resample_rule(draw):
    value = draw(st.integers(min_value=1, max_value=10))
    unit = draw(st.sampled_from(['min', 'h', 'D', 'W', 'M']))
    return f'{value}{unit}'


@given(
    divisions=timestamp_divisions(),
    rule=resample_rule(),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@settings(max_examples=500)
def test_resample_bin_and_out_divs_sorted(divisions, rule, closed, label):
    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)
    except Exception:
        assume(False)

    assert list(outdivs) == sorted(outdivs), \
        f"outdivs not sorted: {outdivs}"
```

**Failing input**: divisions=`DatetimeIndex(['2020-02-29 00:00:00', '2020-02-29 00:01:00'], freq='min')`, rule=`'1M'`, closed=`'right'`, label=`'right'`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

divisions = pd.DatetimeIndex(
    ['2020-02-29 00:00:00', '2020-02-29 00:01:00'],
    dtype='datetime64[ns]',
    freq='min'
)
rule = '1M'
closed = 'right'
label = 'right'

newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)

print(f"outdivs: {outdivs}")
print(f"Sorted? {list(outdivs) == sorted(outdivs)}")
```

Output:
```
outdivs: (Timestamp('2020-02-29 00:00:00'), Timestamp('2020-01-31 00:00:00'))
Sorted? False
```

The output divisions have February before January, clearly violating sort order.

## Why This Is A Bug

1. **Violates fundamental invariant**: Dask dataframe divisions must always be sorted for correct operation
2. **Silent data corruption**: Downstream operations assume sorted divisions, leading to incorrect results
3. **Breaks the contract**: The function returns tuples of timestamps that should be monotonically increasing

The root cause is on line 101 of `resample.py`:

```python
elif outdivs[-1] < divisions[-1]:
    setter(outdivs, temp.index[-1])
```

When `label='right'`, `outdivs = tempdivs + rule` (line 82), shifting all timestamps forward by the rule period. However, `temp.index[-1]` is from the *unshifted* tempdivs, so it can be much earlier than `outdivs[0]`, breaking sort order when assigned to `outdivs[-1]`.

## Fix

The fix should ensure that when adjusting `outdivs[-1]`, the value used maintains sort order. Instead of using `temp.index[-1]` directly, it should be shifted appropriately when `label='right'`:

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -98,7 +98,11 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
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