# Bug Report: dask.dataframe.tseries Division Length Mismatch

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function returns bin divisions and output divisions with mismatched lengths when using `closed='right'` and `label='right'` parameters. This violates the expectation that both tuples should have the same length, as they are used together to define partition boundaries.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

@st.composite
def date_range_strategy(draw):
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))
    start = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    duration_days = draw(st.integers(min_value=1, max_value=365))
    end = start + pd.Timedelta(days=duration_days)
    freq = draw(st.sampled_from(['h', 'D', '2h', '6h', '12h', '2D', '3D']))
    divisions = pd.date_range(start, end, freq=freq)
    assume(len(divisions) >= 2)
    return list(divisions)

@st.composite
def resample_freq_strategy(draw):
    return draw(st.sampled_from(['h', 'D', '2h', '6h', '12h', '2D', '3D', 'W']))

@given(divisions=date_range_strategy(), rule=resample_freq_strategy(),
       closed=st.sampled_from(['left', 'right']),
       label=st.sampled_from(['left', 'right']))
@settings(max_examples=50)
def test_resample_divisions_with_closed_label(divisions, rule, closed, label):
    assume(len(divisions) >= 2)
    try:
        bin_divs, out_divs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)
    except Exception:
        assume(False)
    assert len(bin_divs) == len(out_divs), "Bin and output divisions should have same length"
```

**Failing input**: `divisions` = hourly timestamps from 2000-01-01 00:00:00 to 2000-01-02 00:00:00 (25 timestamps), `rule='D'`, `closed='right'`, `label='right'`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

divisions = [pd.Timestamp('2000-01-01 00:00:00'),
             pd.Timestamp('2000-01-01 01:00:00'),
             pd.Timestamp('2000-01-01 02:00:00'),
             pd.Timestamp('2000-01-01 03:00:00'),
             pd.Timestamp('2000-01-01 04:00:00'),
             pd.Timestamp('2000-01-01 05:00:00'),
             pd.Timestamp('2000-01-01 06:00:00'),
             pd.Timestamp('2000-01-01 07:00:00'),
             pd.Timestamp('2000-01-01 08:00:00'),
             pd.Timestamp('2000-01-01 09:00:00'),
             pd.Timestamp('2000-01-01 10:00:00'),
             pd.Timestamp('2000-01-01 11:00:00'),
             pd.Timestamp('2000-01-01 12:00:00'),
             pd.Timestamp('2000-01-01 13:00:00'),
             pd.Timestamp('2000-01-01 14:00:00'),
             pd.Timestamp('2000-01-01 15:00:00'),
             pd.Timestamp('2000-01-01 16:00:00'),
             pd.Timestamp('2000-01-01 17:00:00'),
             pd.Timestamp('2000-01-01 18:00:00'),
             pd.Timestamp('2000-01-01 19:00:00'),
             pd.Timestamp('2000-01-01 20:00:00'),
             pd.Timestamp('2000-01-01 21:00:00'),
             pd.Timestamp('2000-01-01 22:00:00'),
             pd.Timestamp('2000-01-01 23:00:00'),
             pd.Timestamp('2000-01-02 00:00:00')]

bin_divs, out_divs = _resample_bin_and_out_divs(divisions, 'D', closed='right', label='right')

print(f"Bin divisions ({len(bin_divs)}): {bin_divs}")
print(f"Out divisions ({len(out_divs)}): {out_divs}")

assert len(bin_divs) == len(out_divs), \
    f"Division length mismatch: bin_divs has {len(bin_divs)} elements, out_divs has {len(out_divs)} elements"
```

Output:
```
Bin divisions (3): (Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-01 00:00:00.000000001'), Timestamp('2000-01-02 00:00:00.000000001'))
Out divisions (2): (Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-02 00:00:00'))
AssertionError: Division length mismatch: bin_divs has 3 elements, out_divs has 2 elements
```

## Why This Is A Bug

The function returns two tuples that are meant to be used together to define partition boundaries. The code in `ResampleAggregation._divisions()` (line 193-194 of resample.py) expects them to have the same length:

```python
def _divisions(self):
    return list(self.divisions_left.iterable) + [self.divisions_right.iterable[-1]]
```

When the lengths mismatch, this creates inconsistent partition boundaries. The root cause is in lines 92-101 of `_resample_bin_and_out_divs`:

```python
if newdivs[-1] < divisions[-1]:
    if len(newdivs) < len(divs):
        setter = lambda a, val: a.append(val)
    else:
        setter = lambda a, val: a.__setitem__(-1, val)
    setter(newdivs, divisions[-1] + res)  # Always appends/sets to newdivs
    if outdivs[-1] > divisions[-1]:
        setter(outdivs, outdivs[-1])      # Conditionally updates outdivs
    elif outdivs[-1] < divisions[-1]:
        setter(outdivs, temp.index[-1])   # Conditionally updates outdivs
```

The issue is that `newdivs` is always modified (appended or updated), but `outdivs` is only modified conditionally. When `len(newdivs) < len(divs)` and the conditions for updating `outdivs` are not met, `newdivs` gets an extra element while `outdivs` doesn't.

## Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -92,11 +92,14 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
     if newdivs[-1] < divisions[-1]:
         if len(newdivs) < len(divs):
             setter = lambda a, val: a.append(val)
+            newdivs.append(divisions[-1] + res)
+            outdivs.append(temp.index[-1] if outdivs[-1] < divisions[-1] else outdivs[-1])
         else:
             setter = lambda a, val: a.__setitem__(-1, val)
-        setter(newdivs, divisions[-1] + res)
-        if outdivs[-1] > divisions[-1]:
-            setter(outdivs, outdivs[-1])
-        elif outdivs[-1] < divisions[-1]:
-            setter(outdivs, temp.index[-1])
+            newdivs[-1] = divisions[-1] + res
+            if outdivs[-1] > divisions[-1]:
+                outdivs[-1] = outdivs[-1]
+            elif outdivs[-1] < divisions[-1]:
+                outdivs[-1] = temp.index[-1]

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```