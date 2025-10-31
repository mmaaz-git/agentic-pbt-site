# Bug Report: dask.dataframe.tseries Incorrect Results for Quarterly Resample

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Quarterly (and other non-Tick frequency) resampling with `closed='right'` and `label='right'` returns incorrect results. The last resample bin contains 0 instead of the correct aggregated value, causing silent data corruption.

## Property-Based Test

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np
from hypothesis import given, strategies as st, settings


@given(
    st.integers(min_value=2, max_value=10),
    st.sampled_from(['Q', 'QE', 'M', 'ME']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['sum', 'mean', 'count'])
)
@settings(max_examples=500, deadline=None)
def test_resample_matches_pandas(npartitions, rule, closed, label, method):
    start = pd.Timestamp('2020-01-01')
    end = pd.Timestamp('2020-12-31')

    dates = pd.date_range(start, end, freq='D')
    data = pd.DataFrame({'value': np.arange(len(dates), dtype=float)}, index=dates)

    ddf = dd.from_pandas(data, npartitions=npartitions)

    pandas_result = getattr(data.resample(rule, closed=closed, label=label), method)()
    dask_result = getattr(ddf.resample(rule, closed=closed, label=label), method)().compute()

    pd.testing.assert_frame_equal(
        pandas_result.sort_index(),
        dask_result.sort_index(),
        check_dtype=False,
        atol=1e-10
    )
```

**Failing input**: `npartitions=4, rule='Q', closed='right', label='right', method='sum'`

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np

dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
data = pd.DataFrame({'value': np.arange(len(dates), dtype=float)}, index=dates)

pandas_result = data.resample('Q', closed='right', label='right').sum()
print("Pandas result:")
print(pandas_result)

ddf = dd.from_pandas(data, npartitions=4)
dask_result = ddf.resample('Q', closed='right', label='right').sum().compute()
print("\nDask result:")
print(dask_result)
```

**Expected output (Pandas):**
```
              value
2020-03-31   4095.0
2020-06-30  12376.0
2020-09-30  20930.0
2020-12-31  29394.0
```

**Actual output (Dask):**
```
              value
2020-03-31   4095.0
2020-06-30  12376.0
2020-09-30  20930.0
2020-12-31      0.0  # BUG: Should be 29394.0
```

## Why This Is A Bug

This violates the fundamental contract that Dask should produce the same results as Pandas. The last quarter's aggregated value is completely wrong (0.0 instead of 29394.0), causing silent data corruption. This is a high-severity logic bug because:

1. It returns incorrect results without any warning
2. It affects core resample functionality
3. It only manifests for certain parameter combinations, making it hard to detect
4. Users expect Dask to match Pandas behavior

## Root Cause

The bug is in `_resample_bin_and_out_divs` at lines 93-101. The function returns `newdivs` and `outdivs` with different lengths for non-Tick frequencies (Q, M, Y):

- `newdivs` has 5 elements (used to repartition input data into 4 partitions)
- `outdivs` has 4 elements (used as output divisions for 3 partitions)

This mismatch causes `ResampleReduction._lower()` to create an incorrect number of partitions. The repartitioning creates 4 partitions (from 5 newdivs), but the blockwise operation only processes 3 partitions (from 4 outdivs), leaving the last partition unprocessed.

The root cause is the lambda setter pattern on lines 94-96:
```python
if len(newdivs) < len(divs):
    setter = lambda a, val: a.append(val)
else:
    setter = lambda a, val: a.__setitem__(-1, val)
```

The setter is chosen based on `len(newdivs)` but applied to both `newdivs` and `outdivs`, which may have different lengths. This can cause `newdivs` to be extended (via append) while `outdivs` is not, resulting in the length mismatch.

## Fix

The fix is to ensure `newdivs` and `outdivs` always have the same length, or to use separate setter logic for each array. Here's a patch:

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -88,15 +88,24 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):

     # Adjust ends
     if newdivs[0] < divisions[0]:
         newdivs[0] = divisions[0]
     if newdivs[-1] < divisions[-1]:
-        if len(newdivs) < len(divs):
-            setter = lambda a, val: a.append(val)
+        # Fix newdivs
+        if len(newdivs) < len(divs):
+            newdivs.append(divisions[-1] + res)
         else:
-            setter = lambda a, val: a.__setitem__(-1, val)
-        setter(newdivs, divisions[-1] + res)
+            newdivs[-1] = divisions[-1] + res
+
+        # Fix outdivs separately with its own length check
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

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```