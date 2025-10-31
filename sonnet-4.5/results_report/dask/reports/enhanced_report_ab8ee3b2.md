# Bug Report: dask.dataframe.tseries Quarterly Resample Returns Zero Instead of Correct Sum

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Quarterly and monthly resampling with `closed='right'` and `label='right'` returns 0.0 instead of the correct aggregated value for the last time period, causing silent data corruption.

## Property-Based Test

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np
from hypothesis import given, strategies as st, settings, example
import traceback
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


@given(
    st.integers(min_value=2, max_value=10),
    st.sampled_from(['Q', 'QE', 'M', 'ME']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['sum', 'mean', 'count'])
)
@example(4, 'Q', 'right', 'right', 'sum')  # The specific failing case
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

# Run the test
print("Running property-based test with specific failing example...")
try:
    test_resample_matches_pandas()
    print("All tests passed!")
except AssertionError as e:
    print("Test failed with AssertionError:")
    print(str(e))
    print("\nFailing input: npartitions=4, rule='Q', closed='right', label='right', method='sum'")
except Exception as e:
    print(f"Test failed with {type(e).__name__}:")
    print(traceback.format_exc())
```

<details>

<summary>
**Failing input**: `npartitions=4, rule='Q', closed='right', label='right', method='sum'`
</summary>
```
Running property-based test with specific failing example...
Test failed with AssertionError:
DataFrame.iloc[:, 0] (column name="value") are different

DataFrame.iloc[:, 0] (column name="value") values are different (25.0 %)
[index]: [2020-03-31T00:00:00.000000000, 2020-06-30T00:00:00.000000000, 2020-09-30T00:00:00.000000000, 2020-12-31T00:00:00.000000000]
[left]:  [4095.0, 12376.0, 20930.0, 29394.0]
[right]: [4095.0, 12376.0, 20930.0, 0.0]
At positional index 3, first diff: 29394.0 != 0.0

Failing input: npartitions=4, rule='Q', closed='right', label='right', method='sum'
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np

# Create test data
dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
data = pd.DataFrame({'value': np.arange(len(dates), dtype=float)}, index=dates)

# Pandas result (expected)
pandas_result = data.resample('Q', closed='right', label='right').sum()
print("Pandas result (expected):")
print(pandas_result)
print()

# Dask result (actual)
ddf = dd.from_pandas(data, npartitions=4)
dask_result = ddf.resample('Q', closed='right', label='right').sum().compute()
print("Dask result (actual):")
print(dask_result)
print()

# Show the difference
print("Difference (Pandas - Dask):")
print(pandas_result - dask_result)
```

<details>

<summary>
Dask incorrectly returns 0.0 for the last quarter instead of 29394.0
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/10/repo.py:10: FutureWarning: 'Q' is deprecated and will be removed in a future version, please use 'QE' instead.
  pandas_result = data.resample('Q', closed='right', label='right').sum()
/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py:133: FutureWarning: 'Q' is deprecated and will be removed in a future version, please use 'QE' instead.
  resample = meta_nonempty(self.frame._meta).resample(self.rule, **self.kwargs)
/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py:67: FutureWarning: 'Q' is deprecated and will be removed in a future version, please use 'QE' instead.
  rule = pd.tseries.frequencies.to_offset(rule)
/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py:38: FutureWarning: 'Q' is deprecated and will be removed in a future version, please use 'QE' instead.
  out = getattr(series.resample(rule, **resample_kwargs), how)(
/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py:47: FutureWarning: 'Q' is deprecated and will be removed in a future version, please use 'QE' instead.
  new_index = pd.date_range(
Pandas result (expected):
              value
2020-03-31   4095.0
2020-06-30  12376.0
2020-09-30  20930.0
2020-12-31  29394.0

Dask result (actual):
              value
2020-03-31   4095.0
2020-06-30  12376.0
2020-09-30  20930.0
2020-12-31      0.0

Difference (Pandas - Dask):
              value
2020-03-31      0.0
2020-06-30      0.0
2020-09-30      0.0
2020-12-31  29394.0
```
</details>

## Why This Is A Bug

This violates Dask's fundamental promise of pandas compatibility. The bug causes **silent data corruption** where:

1. **The mathematical result is completely wrong**: Dask returns 0.0 when the correct sum is 29394.0 - a 100% data loss for Q4 2020
2. **No warning or error is raised**: The computation completes "successfully" with incorrect results
3. **Pandas compatibility is broken**: Dask explicitly documents that resample with 'rule', 'closed', and 'label' parameters should match pandas behavior
4. **Common use case is affected**: Quarterly and monthly aggregations are fundamental operations in financial and business analytics

The documentation states that Dask DataFrames provide "a lazy parallel version of pandas that maintains pandas-like behavior" and that "The API is the same. The execution is the same." The resample method documentation is directly copied from pandas with no caveat that this parameter combination produces different results.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py` at lines 93-101 in the `_resample_bin_and_out_divs` function.

The root cause is a logic error where a single `setter` lambda function is chosen based on `len(newdivs)` but then incorrectly applied to both `newdivs` and `outdivs` arrays, which can have different lengths. For non-Tick frequencies (Q, M, Y) with certain parameters:
- `newdivs` gets 5 elements (for repartitioning into 4 partitions)
- `outdivs` gets 4 elements (for output divisions of 3 partitions)

This length mismatch causes the `ResampleReduction._lower()` method to create an incorrect number of partitions - repartitioning creates 4 partitions but the blockwise operation only processes 3, leaving the last partition's data unprocessed and returning 0.

The bug specifically manifests when:
- Using non-Tick frequency rules (Q, QE, M, ME, Y, etc.)
- With `closed='right'` and `label='right'` parameters
- When the data spans multiple resample periods

## Proposed Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -91,14 +91,23 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
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