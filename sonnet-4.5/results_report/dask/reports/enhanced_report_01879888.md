# Bug Report: dask.dataframe.tseries.resample._resample_series Fails with Anchor-Based Frequencies

**Target**: `dask.dataframe.tseries.resample._resample_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`_resample_series` incorrectly raises ValueError when resampling with anchor-based frequencies (ME, QE, YE, W) on data that doesn't span complete calendar periods, even though this is valid in pandas.

## Property-Based Test

```python
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from dask.dataframe.tseries.resample import _resample_series
import pytest

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
@settings(max_examples=50, deadline=None)
def test_resample_series_index_contained_in_new_index(series, rule):
    assume(len(series) >= 3)

    start = series.index[0]
    end = series.index[-1]

    try:
        result = _resample_series(
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
        # If no error, test passes
        assert isinstance(result, pd.Series)
    except ValueError as e:
        if "Index is not contained within new index" in str(e):
            # This is the bug - pandas resample can create indices outside
            # the [start, end] range for anchor-based frequencies
            print(f"\nFailed with rule={rule}")
            print(f"Series index range: {start} to {end}")
            print(f"Series frequency: {pd.infer_freq(series.index) or 'irregular'}")

            # Show what pandas resample creates
            resampled = series.resample(rule).sum()
            print(f"Pandas resample creates indices at: {list(resampled.index)}")

            # Show what date_range creates
            date_range_result = pd.date_range(start, end, freq=rule, inclusive='both')
            print(f"pd.date_range creates: {list(date_range_result)}")

            pytest.fail(f"Index containment violation with rule={rule}, start={start}, end={end}")

# Run the test
if __name__ == "__main__":
    # Run with a specific seed to find failures
    test_resample_series_index_contained_in_new_index()
```

<details>

<summary>
**Failing input**: `series=pd.Series([0.0]*5, index=pd.date_range('2000-01-01', periods=5, freq='h'))`, `rule='W'`
</summary>
```

Failed with rule=W
Series index range: 2003-03-12 00:00:00 to 2003-03-12 18:00:00
Series frequency: 2h
Pandas resample creates indices at: [Timestamp('2003-03-16 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-12 00:00:00 to 2003-03-12 18:00:00
Series frequency: 2h
Pandas resample creates indices at: [Timestamp('2003-03-16 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-12 00:00:00 to 2003-03-12 18:00:00
Series frequency: 2h
Pandas resample creates indices at: [Timestamp('2003-03-16 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-12 00:00:00 to 2003-03-12 18:00:00
Series frequency: 2h
Pandas resample creates indices at: [Timestamp('2003-03-16 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-12 00:00:00 to 2003-03-12 18:00:00
Series frequency: 2h
Pandas resample creates indices at: [Timestamp('2003-03-16 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-03 00:00:00 to 2003-03-03 18:00:00
Series frequency: 2h
Pandas resample creates indices at: [Timestamp('2003-03-09 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-03 00:00:00 to 2003-03-03 18:00:00
Series frequency: 2h
Pandas resample creates indices at: [Timestamp('2003-03-09 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-03 00:00:00 to 2003-03-03 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2003-03-09 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-03 00:00:00 to 2003-03-03 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2003-03-09 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-03 00:00:00 to 2003-03-03 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2003-03-09 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-03 00:00:00 to 2003-03-03 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2003-03-09 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-03 00:00:00 to 2003-03-03 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2003-03-09 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-03 00:00:00 to 2003-03-03 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2003-03-09 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-03 00:00:00 to 2003-03-03 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2003-03-09 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-03 00:00:00 to 2003-03-03 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2003-03-09 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-03 00:00:00 to 2003-03-03 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2003-03-09 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-03 00:00:00 to 2003-03-03 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2003-03-09 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-03 00:00:00 to 2003-03-03 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2003-03-09 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-03-01 00:00:00 to 2003-03-01 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2003-03-02 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2003-01-01 00:00:00 to 2003-01-01 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2003-01-05 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2000-01-01 00:00:00 to 2000-01-01 09:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2000-01-02 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2000-01-01 00:00:00 to 2000-01-01 04:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2000-01-02 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2017-05-12 00:00:00 to 2017-06-07 00:00:00
Series frequency: 2D
Pandas resample creates indices at: [Timestamp('2017-05-14 00:00:00'), Timestamp('2017-05-21 00:00:00'), Timestamp('2017-05-28 00:00:00'), Timestamp('2017-06-04 00:00:00'), Timestamp('2017-06-11 00:00:00')]
pd.date_range creates: [Timestamp('2017-05-14 00:00:00'), Timestamp('2017-05-21 00:00:00'), Timestamp('2017-05-28 00:00:00'), Timestamp('2017-06-04 00:00:00')]

Failed with rule=W
Series index range: 2013-02-08 00:00:00 to 2013-02-08 18:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2013-02-10 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2009-11-14 00:00:00 to 2009-11-14 04:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2009-11-15 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2001-07-27 00:00:00 to 2001-07-27 14:00:00
Series frequency: 2h
Pandas resample creates indices at: [Timestamp('2001-07-29 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2015-03-22 00:00:00 to 2015-03-24 06:00:00
Series frequency: 2h
Pandas resample creates indices at: [Timestamp('2015-03-22 00:00:00'), Timestamp('2015-03-29 00:00:00')]
pd.date_range creates: [Timestamp('2015-03-22 00:00:00')]

Failed with rule=ME
Series index range: 2002-05-19 00:00:00 to 2002-05-20 02:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2002-05-31 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2002-05-19 00:00:00 to 2002-05-20 02:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2002-05-19 00:00:00'), Timestamp('2002-05-26 00:00:00')]
pd.date_range creates: [Timestamp('2002-05-19 00:00:00')]

Failed with rule=W
Series index range: 2009-12-06 00:00:00 to 2010-01-11 00:00:00
Series frequency: 2D
Pandas resample creates indices at: [Timestamp('2009-12-06 00:00:00'), Timestamp('2009-12-13 00:00:00'), Timestamp('2009-12-20 00:00:00'), Timestamp('2009-12-27 00:00:00'), Timestamp('2010-01-03 00:00:00'), Timestamp('2010-01-10 00:00:00'), Timestamp('2010-01-17 00:00:00')]
pd.date_range creates: [Timestamp('2009-12-06 00:00:00'), Timestamp('2009-12-13 00:00:00'), Timestamp('2009-12-20 00:00:00'), Timestamp('2009-12-27 00:00:00'), Timestamp('2010-01-03 00:00:00'), Timestamp('2010-01-10 00:00:00')]

Failed with rule=W
Series index range: 2014-01-17 00:00:00 to 2014-03-14 00:00:00
Series frequency: 2D
Pandas resample creates indices at: [Timestamp('2014-01-19 00:00:00'), Timestamp('2014-01-26 00:00:00'), Timestamp('2014-02-02 00:00:00'), Timestamp('2014-02-09 00:00:00'), Timestamp('2014-02-16 00:00:00'), Timestamp('2014-02-23 00:00:00'), Timestamp('2014-03-02 00:00:00'), Timestamp('2014-03-09 00:00:00'), Timestamp('2014-03-16 00:00:00')]
pd.date_range creates: [Timestamp('2014-01-19 00:00:00'), Timestamp('2014-01-26 00:00:00'), Timestamp('2014-02-02 00:00:00'), Timestamp('2014-02-09 00:00:00'), Timestamp('2014-02-16 00:00:00'), Timestamp('2014-02-23 00:00:00'), Timestamp('2014-03-02 00:00:00'), Timestamp('2014-03-09 00:00:00')]

Failed with rule=W
Series index range: 2001-11-27 00:00:00 to 2001-11-27 14:30:00
Series frequency: 30min
Pandas resample creates indices at: [Timestamp('2001-12-02 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2013-05-25 00:00:00 to 2013-05-25 05:00:00
Series frequency: 30min
Pandas resample creates indices at: [Timestamp('2013-05-26 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2015-08-21 00:00:00 to 2015-08-22 01:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2015-08-23 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2019-08-06 00:00:00 to 2019-08-07 08:00:00
Series frequency: 2h
Pandas resample creates indices at: [Timestamp('2019-08-11 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2010-03-25 00:00:00 to 2010-04-10 00:00:00
Series frequency: D
Pandas resample creates indices at: [Timestamp('2010-03-28 00:00:00'), Timestamp('2010-04-04 00:00:00'), Timestamp('2010-04-11 00:00:00')]
pd.date_range creates: [Timestamp('2010-03-28 00:00:00'), Timestamp('2010-04-04 00:00:00')]

Failed with rule=W
Series index range: 2013-03-27 00:00:00 to 2013-05-24 00:00:00
Series frequency: 2D
Pandas resample creates indices at: [Timestamp('2013-03-31 00:00:00'), Timestamp('2013-04-07 00:00:00'), Timestamp('2013-04-14 00:00:00'), Timestamp('2013-04-21 00:00:00'), Timestamp('2013-04-28 00:00:00'), Timestamp('2013-05-05 00:00:00'), Timestamp('2013-05-12 00:00:00'), Timestamp('2013-05-19 00:00:00'), Timestamp('2013-05-26 00:00:00')]
pd.date_range creates: [Timestamp('2013-03-31 00:00:00'), Timestamp('2013-04-07 00:00:00'), Timestamp('2013-04-14 00:00:00'), Timestamp('2013-04-21 00:00:00'), Timestamp('2013-04-28 00:00:00'), Timestamp('2013-05-05 00:00:00'), Timestamp('2013-05-12 00:00:00'), Timestamp('2013-05-19 00:00:00')]

Failed with rule=W
Series index range: 2005-05-07 00:00:00 to 2005-05-07 16:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2005-05-08 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2000-03-15 00:00:00 to 2000-03-17 00:00:00
Series frequency: 2h
Pandas resample creates indices at: [Timestamp('2000-03-19 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2006-05-13 00:00:00 to 2006-05-13 12:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2006-05-14 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2011-02-02 00:00:00 to 2011-03-03 00:00:00
Series frequency: D
Pandas resample creates indices at: [Timestamp('2011-02-06 00:00:00'), Timestamp('2011-02-13 00:00:00'), Timestamp('2011-02-20 00:00:00'), Timestamp('2011-02-27 00:00:00'), Timestamp('2011-03-06 00:00:00')]
pd.date_range creates: [Timestamp('2011-02-06 00:00:00'), Timestamp('2011-02-13 00:00:00'), Timestamp('2011-02-20 00:00:00'), Timestamp('2011-02-27 00:00:00')]

Failed with rule=W
Series index range: 2019-05-26 00:00:00 to 2019-05-27 00:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2019-05-26 00:00:00'), Timestamp('2019-06-02 00:00:00')]
pd.date_range creates: [Timestamp('2019-05-26 00:00:00')]

Failed with rule=W
Series index range: 2020-04-08 00:00:00 to 2020-04-08 23:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2020-04-12 00:00:00')]
pd.date_range creates: []

Failed with rule=W
Series index range: 2000-01-01 00:00:00 to 2000-01-01 04:00:00
Series frequency: h
Pandas resample creates indices at: [Timestamp('2000-01-02 00:00:00')]
pd.date_range creates: []
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/hypo.py", line 41, in test_resample_series_index_contained_in_new_index
    result = _resample_series(
        series=series,
    ...<8 lines>...
        how_kwargs={}
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py", line 57, in _resample_series
    raise ValueError(
    ...<3 lines>...
    )
ValueError: Index is not contained within new index. This can often be resolved by using larger partitions, or unambiguous frequencies: 'Q', 'A'...

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/hypo.py", line 76, in <module>
    test_resample_series_index_contained_in_new_index()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/hypo.py", line 30, in test_resample_series_index_contained_in_new_index
    time_series_strategy(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/hypo.py", line 71, in test_resample_series_index_contained_in_new_index
    pytest.fail(f"Index containment violation with rule={rule}, start={start}, end={end}")
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: Index containment violation with rule=W, start=2000-01-01 00:00:00, end=2000-01-01 04:00:00
Falsifying example: test_resample_series_index_contained_in_new_index(
    series=2000-01-01 00:00:00    0.0
    2000-01-01 01:00:00    0.0
    2000-01-01 02:00:00    0.0
    2000-01-01 03:00:00    0.0
    2000-01-01 04:00:00    0.0
    Freq: h, dtype: float64,
    rule='W',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/envs/dask_env/hypo.py:65
        /home/npc/pbt/agentic-pbt/envs/dask_env/hypo.py:69
        /home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py:57
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/datetimes.py:655
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/datetimes.py:661
        (and 18 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_series

# Create a simple time series that spans less than a month
series = pd.Series(
    [0.0, 0.0, 0.0, 0.0, 0.0],
    index=pd.date_range('2000-01-01', periods=5, freq='h')
)

print("Series:")
print(series)
print(f"\nSeries index range: {series.index[0]} to {series.index[-1]}")

# Show what pandas resample produces
resampled_pandas = series.resample('ME').sum()
print(f"\nPandas resample('ME').sum() creates index at: {resampled_pandas.index[0]}")
print(f"Pandas resample result:\n{resampled_pandas}")

# Show what pd.date_range would create with same start/end
start = series.index[0]
end = series.index[-1]
expected_range = pd.date_range(start, end, freq='ME', inclusive='both')
print(f"\npd.date_range('{start}', '{end}', freq='ME', inclusive='both') creates:")
print(f"{expected_range.tolist()}")

print("\n" + "="*60)
print("Calling _resample_series with 'ME' frequency...")
print("="*60)

# Try to use _resample_series - this will fail
try:
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
    print(f"Result: {result}")
except ValueError as e:
    print(f"\nValueError: {e}")
```

<details>

<summary>
ValueError: Index is not contained within new index
</summary>
```
Series:
2000-01-01 00:00:00    0.0
2000-01-01 01:00:00    0.0
2000-01-01 02:00:00    0.0
2000-01-01 03:00:00    0.0
2000-01-01 04:00:00    0.0
Freq: h, dtype: float64

Series index range: 2000-01-01 00:00:00 to 2000-01-01 04:00:00

Pandas resample('ME').sum() creates index at: 2000-01-31 00:00:00
Pandas resample result:
2000-01-31    0.0
Freq: ME, dtype: float64

pd.date_range('2000-01-01 00:00:00', '2000-01-01 04:00:00', freq='ME', inclusive='both') creates:
[]

============================================================
Calling _resample_series with 'ME' frequency...
============================================================

ValueError: Index is not contained within new index. This can often be resolved by using larger partitions, or unambiguous frequencies: 'Q', 'A'...
```
</details>

## Why This Is A Bug

The function uses `pd.date_range(start, end, freq=rule)` to construct the expected resampled index, but this is incorrect for anchor-based frequencies. Pandas `resample()` creates indices at anchor points (month-end, quarter-end, week-end) even when these fall outside the original data's exact time range. For example:
- Data from Jan 1 00:00 to Jan 1 04:00 (5 hours) resampled to 'ME' creates an index at Jan 31
- But `pd.date_range('2000-01-01 00:00', '2000-01-01 04:00', freq='ME')` returns an empty DatetimeIndex

This makes dask's resample functionality unusable for common business analytics like monthly, quarterly, or weekly reporting on partial period data.

## Relevant Context

The bug occurs at lines 47-56 of `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py`:

1. Line 38-40: Performs pandas resample correctly and gets the right index
2. Line 47-54: Incorrectly uses `pd.date_range` to reconstruct the index
3. Line 56: Checks if the pandas resample index is contained in the date_range index - fails for anchor frequencies

The error message "This can often be resolved by using larger partitions" is misleading - partition size won't help when the entire dataset spans less than one period.

Documentation: https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.resample.html states that dask resample should work like pandas resample.

## Proposed Fix

The function should use the actual resampled index instead of trying to reconstruct it:

```diff
--- a/resample.py
+++ b/resample.py
@@ -38,26 +38,15 @@ def _resample_series(
     out = getattr(series.resample(rule, **resample_kwargs), how)(
         *how_args, **how_kwargs
     )
-    if reindex_closed is None:
-        inclusive = "both"
-    else:
-        inclusive = reindex_closed
-    closed_kwargs = {"inclusive": inclusive}
-
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
-        raise ValueError(
-            "Index is not contained within new index. This can often be "
-            "resolved by using larger partitions, or unambiguous "
-            "frequencies: 'Q', 'A'..."
-        )
-
-    return out.reindex(new_index, fill_value=fill_value)
+
+    # For anchor-based frequencies, pandas resample can create indices
+    # outside the [start, end] range. Use the actual resample output
+    # index rather than trying to reconstruct it with date_range
+    if fill_value != np.nan:
+        # Only reindex if we need to fill missing values
+        full_range = pd.date_range(out.index.min(), out.index.max(),
+                                   freq=rule, name=out.index.name)
+        return out.reindex(full_range, fill_value=fill_value)
+    else:
+        # Return the resample output as-is
+        return out
```