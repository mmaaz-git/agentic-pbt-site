# Bug Report: dask.dataframe.tseries Division Length Mismatch

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function returns `newdivs` and `outdivs` with mismatched lengths when certain conditions are met, leading to an AssertionError during computation.

## Property-Based Test

```python
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, assume
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@st.composite
def date_range_divisions(draw):
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))

    periods = draw(st.integers(min_value=2, max_value=100))
    freq_choice = draw(st.sampled_from(['h', 'D', '2h', '6h', '12h', '2D', '3D']))

    start = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    divisions = pd.date_range(start=start, periods=periods, freq=freq_choice)

    return list(divisions)


@st.composite
def resample_params(draw):
    divisions = draw(date_range_divisions())

    rule = draw(st.sampled_from(['h', '2h', '6h', '12h', 'D', '2D', '3D', 'W', 'ME']))
    closed = draw(st.sampled_from(['left', 'right']))
    label = draw(st.sampled_from(['left', 'right']))

    return divisions, rule, closed, label


@settings(max_examples=200)
@given(resample_params())
def test_resample_divisions_same_length(params):
    divisions, rule, closed, label = params

    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)
    except Exception:
        assume(False)

    assert len(newdivs) == len(outdivs), f"newdivs and outdivs have different lengths: {len(newdivs)} vs {len(outdivs)}"
```

**Failing input**: `([Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-01 06:00:00'), Timestamp('2000-01-01 12:00:00')], '12h', 'right', 'right')`

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

index = pd.date_range('2000-01-01 00:00:00', periods=3, freq='6h')
series = pd.Series(range(len(index)), index=index)
dask_series = dd.from_pandas(series, npartitions=2)

result = dask_series.resample('12h', closed='right', label='right').count()
computed = result.compute()
```

This raises:
```
AssertionError in /dask/dataframe/dask_expr/_repartition.py:192
assert npartitions_input > npartitions
```

## Why This Is A Bug

The function `_resample_bin_and_out_divs` is supposed to return two division tuples of the same length. The bug occurs in the adjustment logic (lines 89-101 of resample.py):

1. When `newdivs[-1] < divisions[-1]` and `len(newdivs) < len(divs)`, the code uses `append` to add an element to `newdivs`
2. The code then checks `outdivs[-1]` against `divisions[-1]` to decide if `outdivs` should also be appended to
3. When `outdivs[-1] == divisions[-1]`, neither the `>` nor `<` condition is satisfied, so `outdivs` is not modified
4. This results in `newdivs` having one more element than `outdivs`

The length mismatch violates the implicit contract that these two division tuples should have the same length, as they are used together to create partition boundaries. This causes downstream assertion failures when the code tries to repartition the dataframe.

## Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -95,9 +95,11 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         else:
             setter = lambda a, val: a.__setitem__(-1, val)
         setter(newdivs, divisions[-1] + res)
         if outdivs[-1] > divisions[-1]:
             setter(outdivs, outdivs[-1])
         elif outdivs[-1] < divisions[-1]:
             setter(outdivs, temp.index[-1])
+        else:
+            setter(outdivs, outdivs[-1])

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```