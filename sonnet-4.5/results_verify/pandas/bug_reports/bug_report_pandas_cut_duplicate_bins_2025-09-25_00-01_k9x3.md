# Bug Report: pandas.core.reshape.tile.cut - Duplicate Bins from Small Value Ranges

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.cut()` crashes with a `ValueError` about duplicate bins when called with an integer number of bins on arrays with extremely small value ranges. The underlying issue is that `np.linspace()` produces duplicate values when the range is smaller than floating-point precision allows, but `pd.cut()` doesn't automatically apply the `duplicates='drop'` behavior that users might expect.

## Property-Based Test

```python
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings

@given(
    data=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100),
    bins=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=500)
def test_cut_bin_count(data, bins):
    arr = np.array(data)
    result = pd.cut(arr, bins)
    assert len(result) == len(arr)
```

**Failing input**: `data=[0.0, 5e-324], bins=2`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

arr = np.array([0.0, 5e-324])
result, bins = pd.cut(arr, 2, retbins=True)
```

**Output**:
```
ValueError: Bin edges must be unique: Index([0.0, 0.0, 5e-324], dtype='float64').
You can drop duplicate edges by setting the 'duplicates' kwarg
```

## Why This Is A Bug

While the error message is informative, the behavior is inconsistent:

1. When users pass `bins` as an integer, they're asking pandas to automatically determine appropriate bin edges
2. For extremely small ranges (below floating-point precision), `np.linspace()` mathematically cannot create the requested number of distinct bins
3. The function requires users to know to use `duplicates='drop'`, but this parameter is not well-documented for the integer bins case
4. Users passing valid numeric data shouldn't need to handle floating-point precision edge cases manually

The function should either:
- Automatically handle this case (preferred), or
- Raise a clearer error message explaining the precision issue

## Root Cause

In `_nbins_to_bins()` at lines 403-408:

```python
bins = np.linspace(mn, mx, nbins + 1, endpoint=True)
adj = (mx - mn) * 0.001  # 0.1% of the range
if right:
    bins[0] -= adj
else:
    bins[-1] += adj
```

When `mx - mn` is smaller than ~1e-15, the adjustment `adj` becomes so small that `bins[0] - adj` equals `bins[0]` due to floating-point precision limits, creating duplicates.

## Fix

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -245,7 +245,7 @@ def cut(
     if not np.iterable(bins):
         bins = _nbins_to_bins(x_idx, bins, right)
-
+        duplicates_from_precision = True
     elif isinstance(bins, IntervalIndex):
         if bins.is_overlapping:
             raise ValueError("Overlapping IntervalIndex is not accepted.")
@@ -253,6 +253,7 @@ def cut(
     else:
         bins = Index(bins)
         if not bins.is_monotonic_increasing:
             raise ValueError("bins must increase monotonically.")
+        duplicates_from_precision = False

     fac, bins = _bins_to_cuts(
         x_idx,
@@ -261,7 +262,7 @@ def cut(
         labels=labels,
         precision=precision,
         include_lowest=include_lowest,
-        duplicates=duplicates,
+        duplicates='drop' if duplicates_from_precision else duplicates,
         ordered=ordered,
     )
```

Alternatively, detect this case in `_nbins_to_bins()` and raise a more informative error:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -389,6 +389,11 @@ def _nbins_to_bins(x_idx: Index, nbins: int, right: bool) -> Index:
             bins = np.linspace(mn, mx, nbins + 1, endpoint=True)
     else:  # adjust end points after binning
         if _is_dt_or_td(x_idx.dtype):
             # ... existing datetime handling
@@ -402,6 +407,13 @@ def _nbins_to_bins(x_idx: Index, nbins: int, right: bool) -> Index:
         else:
             bins = np.linspace(mn, mx, nbins + 1, endpoint=True)
+        # Check if linspace created duplicates due to precision limits
+        if len(np.unique(bins)) < len(bins):
+            raise ValueError(
+                f"Cannot create {nbins} distinct bins for range [{mn}, {mx}]. "
+                f"The value range is too small relative to floating-point precision. "
+                f"Try using fewer bins or scaling your data."
+            )
         adj = (mx - mn) * 0.001  # 0.1% of the range
         if right:
             bins[0] -= adj
```