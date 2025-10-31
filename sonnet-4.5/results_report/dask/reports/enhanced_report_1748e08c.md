# Bug Report: dask.dataframe.tseries.resample Division Length Mismatch

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function returns bin divisions and output divisions with mismatched lengths when using `closed='right'` and `label='right'` parameters, causing assertion errors in downstream processing.

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
    assert len(bin_divs) == len(out_divs), f"Bin and output divisions should have same length: bin_divs has {len(bin_divs)} elements, out_divs has {len(out_divs)} elements"

if __name__ == '__main__':
    test_resample_divisions_with_closed_label()
```

<details>

<summary>
**Failing input**: `divisions=[Timestamp('2000-01-01 00:00:00'), ..., Timestamp('2000-01-02 00:00:00')], rule='D', closed='right', label='right'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 35, in <module>
    test_resample_divisions_with_closed_label()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 23, in test_resample_divisions_with_closed_label
    closed=st.sampled_from(['left', 'right']),
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 32, in test_resample_divisions_with_closed_label
    assert len(bin_divs) == len(out_divs), f"Bin and output divisions should have same length: bin_divs has {len(bin_divs)} elements, out_divs has {len(out_divs)} elements"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Bin and output divisions should have same length: bin_divs has 3 elements, out_divs has 2 elements
Falsifying example: test_resample_divisions_with_closed_label(
    divisions=[Timestamp('2000-01-01 00:00:00'),
     Timestamp('2000-01-01 01:00:00'),
     Timestamp('2000-01-01 02:00:00'),
     Timestamp('2000-01-01 03:00:00'),
     Timestamp('2000-01-01 04:00:00'),
     Timestamp('2000-01-01 05:00:00'),
     Timestamp('2000-01-01 06:00:00'),
     Timestamp('2000-01-01 07:00:00'),
     Timestamp('2000-01-01 08:00:00'),
     Timestamp('2000-01-01 09:00:00'),
     Timestamp('2000-01-01 10:00:00'),
     Timestamp('2000-01-01 11:00:00'),
     Timestamp('2000-01-01 12:00:00'),
     Timestamp('2000-01-01 13:00:00'),
     Timestamp('2000-01-01 14:00:00'),
     Timestamp('2000-01-01 15:00:00'),
     Timestamp('2000-01-01 16:00:00'),
     Timestamp('2000-01-01 17:00:00'),
     Timestamp('2000-01-01 18:00:00'),
     Timestamp('2000-01-01 19:00:00'),
     Timestamp('2000-01-01 20:00:00'),
     Timestamp('2000-01-01 21:00:00'),
     Timestamp('2000-01-01 22:00:00'),
     Timestamp('2000-01-01 23:00:00'),
     Timestamp('2000-01-02 00:00:00')],
    rule='D',
    closed='right',
    label='right',
)
```
</details>

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

<details>

<summary>
AssertionError: Division length mismatch
</summary>
```
Bin divisions (3): (Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-01 00:00:00.000000001'), Timestamp('2000-01-02 00:00:00.000000001'))
Out divisions (2): (Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-02 00:00:00'))
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/repo.py", line 35, in <module>
    assert len(bin_divs) == len(out_divs), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Division length mismatch: bin_divs has 3 elements, out_divs has 2 elements
```
</details>

## Why This Is A Bug

The `_resample_bin_and_out_divs` function is used internally by Dask's resampling functionality to determine partition boundaries for time series data. The function returns two tuples:
1. `bin_divs`: The bin boundaries for grouping data
2. `out_divs`: The output division boundaries after resampling

These two tuples are used together in `ResampleAggregation._lower()` (lines 162-164 of resample.py) to create BlockwiseDep objects that define how data partitions should be processed. The code creates:
- `BlockwiseDep(output_divisions[:-1])`
- `BlockwiseDep(output_divisions[1:])`

When these tuples have different lengths, it creates inconsistent partition boundaries that can lead to data processing errors.

The bug occurs specifically when both `closed='right'` and `label='right'` are used together. The root cause is in the conditional logic at lines 92-101 where `newdivs` is always modified (either appended or updated) but `outdivs` is only modified conditionally based on comparisons with `divisions[-1]`.

## Relevant Context

The parameters `closed='right'` and `label='right'` are standard pandas resample parameters:
- `closed='right'` indicates that the right bin edge is included in each bin
- `label='right'` indicates that the resulting aggregated data should be labeled with the right edge of the bin

These are legitimate parameter combinations that users would expect to work correctly for time series analysis. The bug only manifests with this specific combination - all other combinations of closed/label parameters work correctly.

The function is called from `ResampleReduction._resample_divisions` property (lines 146-149), which is then used throughout the resampling pipeline. While `_resample_bin_and_out_divs` is an internal function (prefixed with underscore), it's a critical component of the public resample API that users interact with.

Documentation reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html

## Proposed Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -90,14 +90,18 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
     if newdivs[0] < divisions[0]:
         newdivs[0] = divisions[0]
     if newdivs[-1] < divisions[-1]:
         if len(newdivs) < len(divs):
-            setter = lambda a, val: a.append(val)
+            # When appending, we need to append to both lists
+            newdivs.append(divisions[-1] + res)
+            if outdivs[-1] < divisions[-1]:
+                outdivs.append(temp.index[-1])
+            else:
+                outdivs.append(outdivs[-1])
         else:
-            setter = lambda a, val: a.__setitem__(-1, val)
-        setter(newdivs, divisions[-1] + res)
-        if outdivs[-1] > divisions[-1]:
-            setter(outdivs, outdivs[-1])
-        elif outdivs[-1] < divisions[-1]:
-            setter(outdivs, temp.index[-1])
+            # When updating last element, update both
+            newdivs[-1] = divisions[-1] + res
+            if outdivs[-1] < divisions[-1]:
+                outdivs[-1] = temp.index[-1]
+            # else keep outdivs[-1] as is

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```