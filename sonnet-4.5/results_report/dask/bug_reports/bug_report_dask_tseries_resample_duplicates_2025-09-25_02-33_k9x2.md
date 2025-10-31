# Bug Report: dask.dataframe.tseries.resample Duplicate Timestamps in Output Divisions

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function produces duplicate timestamps in the `outdivs` output when certain combinations of input parameters are used. Specifically, when the number of output divisions is less than the number of input divisions and `outdivs[-1] < divisions[-1]`, the function appends `temp.index[-1]` to outdivs even though it may already be the last element, creating a duplicate.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, assume, settings
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@given(
    st.integers(min_value=3, max_value=100),
    st.sampled_from(['h', 'D', '2h', '3h', '6h', '12h', '30min', 'W', '2D', '3D', '5D', '7D', '10D']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right'])
)
@settings(max_examples=5000, deadline=None)
def test_outdivs_no_consecutive_duplicates(n_divisions, rule, closed, label):
    start = pd.Timestamp('2020-01-01')
    end = pd.Timestamp('2021-12-31')
    divisions = pd.date_range(start, end, periods=n_divisions)

    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)
    except Exception:
        assume(False)

    for i in range(len(outdivs) - 1):
        if outdivs[i] == outdivs[i + 1]:
            assert False, f"Found consecutive duplicate in outdivs at index {i}: {outdivs[i]}"
```

**Failing input**: `n_divisions=88, rule='10D', closed='right', label='left'`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

start = pd.Timestamp('2020-01-01')
end = pd.Timestamp('2021-12-31')
divisions = pd.date_range(start, end, periods=88)

newdivs, outdivs = _resample_bin_and_out_divs(divisions, '10D', closed='right', label='left')

print(f"outdivs[-5:] = {outdivs[-5:]}")

for i in range(len(outdivs) - 1):
    if outdivs[i] == outdivs[i + 1]:
        print(f"DUPLICATE at index {i}: {outdivs[i]}")
```

Output:
```
outdivs[-5:] = (Timestamp('2021-11-21 00:00:00'), Timestamp('2021-12-01 00:00:00'), Timestamp('2021-12-11 00:00:00'), Timestamp('2021-12-21 00:00:00'), Timestamp('2021-12-21 00:00:00'))
DUPLICATE at index 73: 2021-12-21 00:00:00
```

## Why This Is A Bug

The function `_resample_bin_and_out_divs` is responsible for computing bin divisions and output divisions for time series resampling operations. The output divisions (`outdivs`) should be a monotonically increasing sequence of unique timestamps that define the boundaries of resampled data. Duplicate values in `outdivs` violate this invariant and can lead to incorrect behavior in downstream operations.

The bug occurs in the "Adjust ends" section of the code (lines 33-39 in the original source). When `len(newdivs) < len(divs)`, the code uses `append` to add elements. In the case where `outdivs[-1] < divisions[-1]`, it appends `temp.index[-1]`, but this value is already equal to `outdivs[-1]`, resulting in a duplicate.

## Fix

The issue is that when using `append` (i.e., when `len(newdivs) < len(divs)`), the code should only append to `outdivs` if the value to be appended is different from the current last element. The fix is to check whether the value being appended would create a duplicate before performing the append operation:

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -35,7 +35,7 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         setter = lambda a, val: a.__setitem__(-1, val)
     setter(newdivs, divisions[-1] + res)
     if outdivs[-1] > divisions[-1]:
-        setter(outdivs, outdivs[-1])
+        pass  # outdivs already extends beyond divisions, no adjustment needed
     elif outdivs[-1] < divisions[-1]:
-        setter(outdivs, temp.index[-1])
+        if len(newdivs) >= len(divs) or outdivs[-1] != temp.index[-1]:
+            setter(outdivs, temp.index[-1])
```

Alternative fix using a conditional check:
```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -35,7 +35,10 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         setter = lambda a, val: a.__setitem__(-1, val)
     setter(newdivs, divisions[-1] + res)
     if outdivs[-1] > divisions[-1]:
-        setter(outdivs, outdivs[-1])
+        if len(newdivs) >= len(divs):
+            setter(outdivs, outdivs[-1])
     elif outdivs[-1] < divisions[-1]:
-        setter(outdivs, temp.index[-1])
+        if outdivs[-1] != temp.index[-1]:
+            setter(outdivs, temp.index[-1])
```