# Bug Report: dask.dataframe.tseries._resample_bin_and_out_divs Length Mismatch

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function can return `newdivs` and `outdivs` tuples with different lengths, violating a critical assumption made by `ResampleAggregation._divisions()` which expects them to have the same length.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

@given(
    st.integers(min_value=3, max_value=50),
    st.sampled_from(['h', 'D', '2h', '6h', '12h']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right'])
)
@settings(max_examples=1000)
def test_resample_divisions_with_gaps(n_segments, rule, closed, label):
    segments = []
    start_dates = ['2020-01-01', '2020-02-01']

    for start_str in start_dates:
        start = pd.Timestamp(start_str)
        end = start + pd.Timedelta(days=7)
        segment = pd.date_range(start, end, periods=n_segments // 2)
        segments.append(segment)

    divisions = segments[0].union(segments[1])

    newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)

    assert len(newdivs) == len(outdivs), f"Length mismatch: newdivs={len(newdivs)}, outdivs={len(outdivs)}"
```

**Failing input**: `n_segments=18, rule='D', closed='right', label='right'`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

segments = []
for start_str in ['2020-01-01', '2020-02-01']:
    start = pd.Timestamp(start_str)
    end = start + pd.Timedelta(days=7)
    segment = pd.date_range(start, end, periods=9)
    segments.append(segment)

divisions = segments[0].union(segments[1])

newdivs, outdivs = _resample_bin_and_out_divs(divisions, 'D', closed='right', label='right')

print(f"len(newdivs) = {len(newdivs)}")
print(f"len(outdivs) = {len(outdivs)}")
```

Output:
```
len(newdivs) = 17
len(outdivs) = 16
```

## Why This Is A Bug

The `ResampleAggregation._divisions()` method (line 193-194 in resample.py) assumes that `divisions_left` and `divisions_right` have the same length:

```python
def _divisions(self):
    return list(self.divisions_left.iterable) + [self.divisions_right.iterable[-1]]
```

This method constructs the final divisions by concatenating all elements from `divisions_left` with the last element of `divisions_right`. This only works correctly if both tuples have the same length, which is what `_resample_bin_and_out_divs` is supposed to guarantee.

When the lengths differ, the resulting divisions will be incorrect, leading to data corruption or crashes during resampling operations.

## Fix

The root cause is in lines 90-101 of `resample.py`. The `setter` function is called once on `newdivs` (line 97) but may not be called on `outdivs` if `outdivs[-1] == divisions[-1]`. The fix ensures both arrays are updated consistently:

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

The fix adds an `else` clause to ensure that when `outdivs[-1] == divisions[-1]`, the setter is still called on `outdivs` to maintain length consistency with `newdivs`.