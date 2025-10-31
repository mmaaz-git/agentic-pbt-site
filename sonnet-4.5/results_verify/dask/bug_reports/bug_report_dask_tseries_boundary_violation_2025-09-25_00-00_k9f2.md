# Bug Report: dask.dataframe.tseries Output Divisions Boundary Violation

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function can produce output divisions (`outdivs`) where the first element is before the first input division, extending the time range beyond the input data boundaries.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@given(
    st.integers(min_value=2, max_value=20),
    st.sampled_from(['h', 'D', '2h', '3D', '12h', 'W']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right']),
)
@settings(max_examples=500)
def test_resample_divisions_contain_original_boundaries(n_divs, freq, closed, label):
    start = pd.Timestamp('2000-01-01')
    end = start + pd.Timedelta(days=30)
    divisions = pd.date_range(start, end, periods=n_divs)

    newdivs, outdivs = _resample_bin_and_out_divs(divisions, freq, closed=closed, label=label)

    assert outdivs[0] >= divisions[0], f"First outdiv {outdivs[0]} before first division {divisions[0]}"
```

**Failing input**: `n_divs=2, freq='h', closed='right', label='left'`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

start = pd.Timestamp('2000-01-01')
end = start + pd.Timedelta(days=30)
divisions = pd.date_range(start, end, periods=2)

newdivs, outdivs = _resample_bin_and_out_divs(divisions, 'h', closed='right', label='left')

print(f"Input divisions[0]: {divisions[0]}")
print(f"Output outdivs[0]: {outdivs[0]}")

if outdivs[0] < divisions[0]:
    print(f"ERROR: outdivs[0] is before divisions[0]")
    print(f"  {outdivs[0]} < {divisions[0]}")
```

## Why This Is A Bug

The function adjusts `newdivs[0]` to ensure it doesn't extend before the input range (lines 90-91), but it fails to apply the same constraint to `outdivs[0]`. When `closed='right'`, the code adds a `res` offset to tempdivs (line 78), which can shift `outdivs` to start before the actual data range.

This causes the output divisions to claim data exists before the input time range begins, which can lead to:

1. Incorrect time range queries returning empty or incorrect results
2. Confusion about the actual time span of the resampled data
3. Potential errors when operations expect divisions to match data boundaries

## Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -89,6 +89,8 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
     # Adjust ends
     if newdivs[0] < divisions[0]:
         newdivs[0] = divisions[0]
+    if outdivs[0] < divisions[0]:
+        outdivs[0] = divisions[0]
     if newdivs[-1] < divisions[-1]:
         if len(newdivs) < len(divs):
             setter = lambda a, val: a.append(val)
```

The fix ensures that `outdivs[0]` is also constrained to be at or after `divisions[0]`, matching the boundary preservation logic already applied to `newdivs[0]`.