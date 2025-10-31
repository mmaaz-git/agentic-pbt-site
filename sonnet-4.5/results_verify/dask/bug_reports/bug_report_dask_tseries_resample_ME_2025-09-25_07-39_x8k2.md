# Bug Report: dask.dataframe.tseries.resample _resample_series Month-End Frequency

**Target**: `dask.dataframe.tseries.resample._resample_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`_resample_series` fails with ValueError when resampling with month-end frequency ('ME') or other anchor-based frequencies on data that doesn't span a complete period. The function incorrectly assumes that `pd.date_range(start, end, freq)` will contain all indices created by `series.resample(freq)`, but pandas resample can create indices outside the [start, end] range for anchor-based frequencies.

## Property-Based Test

```python
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from dask.dataframe.tseries.resample import _resample_series

@st.composite
def time_series_strategy(draw):
    n = draw(st.integers(min_value=5, max_value=30))
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))

    try:
        start = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    except:
        assume(False)

    freq = draw(st.sampled_from(['h', '30min', '2h', 'D', '2D']))
    index = pd.date_range(start, periods=n, freq=freq)
    values = draw(st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=n,
        max_size=n
    ))

    return pd.Series(values, index=index)

@given(
    time_series_strategy(),
    st.sampled_from(['D', '2D', 'W', 'h', '2h', '6h', '12h', '30min', 'ME', '3D'])
)
@settings(max_examples=2000, deadline=None)
def test_resample_series_index_contained_in_new_index(series, rule):
    assume(len(series) >= 3)

    start = series.index[0]
    end = series.index[-1]

    try:
        _resample_series(
            series=series,
            start=start,
            end=end,
            reindex_closed=None,
            rule=rule,
            resample_kwargs={},
            how='sum',
            fill_value=0,
            how_args=(),
            how_kwargs={}
        )
    except ValueError as e:
        if "Index is not contained within new index" in str(e):
            pytest.fail(f"Index containment violation with rule={rule}")
```

**Failing input**: `series=pd.Series([0.0]*5, index=pd.date_range('2000-01-01', periods=5, freq='h'))`, `rule='ME'`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_series

series = pd.Series(
    [0.0, 0.0, 0.0, 0.0, 0.0],
    index=pd.date_range('2000-01-01', periods=5, freq='h')
)

start = series.index[0]
end = series.index[-1]

resampled_pandas = series.resample('ME').sum()
print(f"Pandas resample creates index at: {resampled_pandas.index[0]}")

expected_range = pd.date_range(start, end, freq='ME', inclusive='both')
print(f"pd.date_range creates: {expected_range.tolist()}")

result = _resample_series(
    series=series,
    start=start,
    end=end,
    reindex_closed=None,
    rule='ME',
    resample_kwargs={},
    how='sum',
    fill_value=0,
    how_args=(),
    how_kwargs={}
)
```

Output:
```
Pandas resample creates index at: 2000-01-31 00:00:00
pd.date_range creates: []
ValueError: Index is not contained within new index. This can often be resolved by using larger partitions, or unambiguous frequencies: 'Q', 'A'...
```

## Why This Is A Bug

The function uses `pd.date_range(start, end, freq=rule)` (line 47-54) to create the expected index for reindexing. However, when using anchor-based frequencies like 'ME' (month end), 'QE' (quarter end), or 'YE' (year end), pandas' `resample()` creates indices at the anchor points (e.g., month ends) which may fall outside the [start, end] range.

For example:
- Data from 2000-01-01 to 2000-01-01 04:00 (5 hours)
- `series.resample('ME').sum()` creates an index at 2000-01-31 (the month end)
- `pd.date_range('2000-01-01', '2000-01-01 04:00', freq='ME')` returns an empty DatetimeIndex because no month-end falls within this range
- The check on line 56 `if not out.index.isin(new_index).all()` fails and raises ValueError

This makes `_resample_series` unusable with common frequencies like monthly, quarterly, or annual resampling on data that doesn't span complete periods.

## Fix

The issue is that `new_index` should be constructed to match what pandas `resample().agg()` actually creates, not what `date_range` creates. The fix requires using the actual resampled index from the output, or adjusting the date_range to include anchor points.

One approach:

```diff
--- a/resample.py
+++ b/resample.py
@@ -38,6 +38,7 @@ def _resample_series(
     out = getattr(series.resample(rule, **resample_kwargs), how)(
         *how_args, **how_kwargs
     )
+
     if reindex_closed is None:
         inclusive = "both"
     else:
@@ -45,7 +46,7 @@ def _resample_series(
     closed_kwargs = {"inclusive": inclusive}

     new_index = pd.date_range(
-        start.tz_localize(None),
-        end.tz_localize(None),
+        out.index.min().tz_localize(None),
+        out.index.max().tz_localize(None),
         freq=rule,
         **closed_kwargs,
```

This uses the actual min/max from the resampled output to create the date range, ensuring all resampled indices are included.

Alternatively, the function should use the output index directly instead of reconstructing it with `date_range`:

```diff
--- a/resample.py
+++ b/resample.py
@@ -45,17 +45,10 @@ def _resample_series(
     closed_kwargs = {"inclusive": inclusive}

-    new_index = pd.date_range(
-        start.tz_localize(None),
-        end.tz_localize(None),
-        freq=rule,
-        **closed_kwargs,
-        name=out.index.name,
-        unit=out.index.unit,
-    ).tz_localize(start.tz, nonexistent="shift_forward")
-
-    if not out.index.isin(new_index).all():
-        raise ValueError(...)
-
-    return out.reindex(new_index, fill_value=fill_value)
+    if start.tz:
+        new_index = out.index.tz_convert(start.tz)
+    else:
+        new_index = out.index
+
+    return out.reindex(new_index, fill_value=fill_value)
```

However, this may require understanding the original intent of the reindex operation and whether it's needed at all.