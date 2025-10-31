# Bug Report: dask.dataframe.tseries.resample Non-Monotonic Output

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function returns `outdivs` with duplicate timestamps, violating the fundamental invariant that timestamp divisions should be monotonically increasing.

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
def test_resample_bin_and_out_divs_monotonic_increasing(divisions, rule, closed, label):
    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)
    except Exception:
        assume(False)

    for i in range(len(outdivs) - 1):
        assert outdivs[i] < outdivs[i+1], \
            f"outdivs should be monotonically increasing at index {i}: {outdivs[i]} >= {outdivs[i+1]}"
```

**Failing input**: `divisions=(Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-01 01:00:00')), rule='D', closed='left', label='left'`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

divisions = (
    pd.Timestamp('2000-01-01 00:00:00'),
    pd.Timestamp('2000-01-01 01:00:00')
)
rule = 'D'
closed = 'left'
label = 'left'

newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)

print(f"outdivs: {outdivs}")
print(f"outdivs[0]: {outdivs[0]}")
print(f"outdivs[1]: {outdivs[1]}")
print(f"Are they equal? {outdivs[0] == outdivs[1]}")
```

## Why This Is A Bug

Timestamp divisions in Dask are expected to be strictly monotonically increasing to represent ordered partitions. When `outdivs` contains duplicate timestamps, it violates this invariant and can cause downstream processing errors or incorrect results when partitions are accessed by timestamp ranges.

The root cause is in lines 98-101 of `resample.py`. When the setter is configured to use `__setitem__(-1)` instead of `append`, and then `setter(outdivs, outdivs[-1])` is called on line 99, it replaces the last element with itself. However, if `outdivs` initially had only 1 element, the code path at line 101 (`setter(outdivs, temp.index[-1])`) gets executed instead. When there's only 1 element in `temp.index`, `temp.index[-1]` is the same as `outdivs[-1]`, creating a duplicate when the setter appends instead of replacing.

## Fix

The fix requires ensuring that when values are being set, they maintain strict monotonicity. A comprehensive fix would need to reconsider the logic in lines 92-101 to ensure that:
1. Both `newdivs` and `outdivs` always have the same length
2. Both are strictly monotonically increasing
3. The appropriate setter (append vs setitem) is chosen independently for each list based on their respective lengths

A minimal patch addressing both bugs:

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -89,14 +89,22 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
     # Adjust ends
     if newdivs[0] < divisions[0]:
         newdivs[0] = divisions[0]
     if newdivs[-1] < divisions[-1]:
-        if len(newdivs) < len(divs):
-            setter = lambda a, val: a.append(val)
+        if len(newdivs) < len(divs):
+            newdivs.append(divisions[-1] + res)
         else:
-            setter = lambda a, val: a.__setitem__(-1, val)
-        setter(newdivs, divisions[-1] + res)
+            newdivs[-1] = divisions[-1] + res
+
         if outdivs[-1] > divisions[-1]:
-            setter(outdivs, outdivs[-1])
+            if len(outdivs) < len(divs):
+                outdivs.append(outdivs[-1])
+            else:
+                outdivs[-1] = outdivs[-1]
         elif outdivs[-1] < divisions[-1]:
-            setter(outdivs, temp.index[-1])
+            if len(outdivs) < len(divs):
+                outdivs.append(temp.index[-1])
+            else:
+                outdivs[-1] = temp.index[-1]
+        else:
+            if len(outdivs) < len(divs):
+                outdivs.append(outdivs[-1])
```