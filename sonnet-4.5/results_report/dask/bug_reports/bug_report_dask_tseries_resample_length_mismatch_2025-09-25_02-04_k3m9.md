# Bug Report: dask.dataframe.tseries.resample Length Mismatch

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function returns `newdivs` and `outdivs` with different lengths, violating the invariant that these two tuples should always have the same length since they are used together as start/end pairs in downstream processing.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@st.composite
def date_range_divisions(draw):
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))
    num_periods = draw(st.integers(min_value=2, max_value=100))
    freq = draw(st.sampled_from(['h', 'D', '2h', '30min', 'W']))
    start = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    divisions = pd.date_range(start=start, periods=num_periods, freq=freq)
    return tuple(divisions)


@given(
    divisions=date_range_divisions(),
    rule=st.sampled_from(['h', 'D', '2D', '30min', 'W', '2W']),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@settings(max_examples=500)
def test_resample_bin_and_out_divs_output_lengths_match(divisions, rule, closed, label):
    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)
    except Exception:
        assume(False)

    assert len(newdivs) == len(outdivs), \
        f"newdivs and outdivs should have same length: {len(newdivs)} != {len(outdivs)}"
```

**Failing input**: `divisions=(Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-02 00:00:00'), Timestamp('2000-01-03 00:00:00')), rule='2D', closed='right', label='right'`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

divisions = (
    pd.Timestamp('2000-01-01 00:00:00'),
    pd.Timestamp('2000-01-02 00:00:00'),
    pd.Timestamp('2000-01-03 00:00:00')
)
rule = '2D'
closed = 'right'
label = 'right'

newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)

print(f"newdivs length: {len(newdivs)}")
print(f"outdivs length: {len(outdivs)}")
print(f"newdivs: {newdivs}")
print(f"outdivs: {outdivs}")
```

## Why This Is A Bug

The function is supposed to return two tuples of equal length representing corresponding bin divisions and output divisions. These are used together in `ResampleAggregation` where `divisions_left.iterable` and `divisions_right.iterable` are accessed with the same indices. When the lengths mismatch, this causes index out of bounds errors or incorrect pairing of start/end timestamps.

The root cause is in lines 92-101 of `resample.py`. When `newdivs[-1] < divisions[-1]`, the code appends a value to `newdivs` (line 97). However, it only modifies `outdivs` if `outdivs[-1] != divisions[-1]` (lines 98-101). When `outdivs[-1] == divisions[-1]`, `outdivs` is not modified, creating a length mismatch.

## Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -96,6 +96,8 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
             setter = lambda a, val: a.__setitem__(-1, val)
         setter(newdivs, divisions[-1] + res)
         if outdivs[-1] > divisions[-1]:
             setter(outdivs, outdivs[-1])
         elif outdivs[-1] < divisions[-1]:
             setter(outdivs, temp.index[-1])
+        else:
+            setter(outdivs, outdivs[-1])
```