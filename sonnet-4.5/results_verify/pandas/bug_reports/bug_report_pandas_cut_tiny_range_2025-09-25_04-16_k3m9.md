# Bug Report: pandas.core.reshape.tile.cut Silently Returns All NaN for Tiny Data Ranges

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pd.cut()` silently returns all NaN values with 0 categories when the data range is extremely small (on the order of 1e-308), instead of either binning the data or raising an informative error.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd

@given(
    data=st.lists(st.floats(min_value=0, max_value=100, allow_nan=False), min_size=10, max_size=100),
    bins=st.integers(min_value=2, max_value=10)
)
def test_cut_all_values_binned(data, bins):
    result = pd.cut(data, bins=bins, duplicates='drop')
    non_null_count = result.notna().sum()
    assert non_null_count == len(data), f"Not all values were binned: {non_null_count}/{len(data)}"
```

**Failing input**: `data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1125369292536007e-308], bins=2`

## Reproducing the Bug

```python
import pandas as pd

data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1125369292536007e-308]
result = pd.cut(data, bins=2)

print(f"Result: {result}")
print(f"Categories: {result.categories}")
print(f"Non-null count: {result.notna().sum()} / {len(data)}")
```

Output:
```
Result: [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
Categories: IntervalIndex([], dtype='interval[float64, right]')
Non-null count: 0 / 10
```

## Why This Is A Bug

The function `pd.cut()` is designed to "bin values into discrete intervals." Users reasonably expect that:
1. All non-NaN input values will be assigned to a bin
2. If binning is impossible, an informative error should be raised

Instead, when the data range is very small (< ~1e-300), the function silently returns all NaN values with 0 categories. This violates the principle of least surprise and makes debugging difficult.

The issue appears to be related to floating-point precision: when computing bin edges for an extremely small range, numerical errors cause all bin edges to collapse to the same value, which then get dropped as duplicates.

## Fix

The fix should detect when bin edge computation results in insufficient unique bins and either:
1. Raise a `ValueError` with a clear message like "Cannot create bins: data range too small for float64 precision"
2. Fall back to creating a single bin containing all values

A patch might look like:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -440,6 +440,11 @@ def _bins_to_cuts(
     if duplicates == "drop":
         bins = bins[adj.nonzero()[0]]

+    # Check if we have enough bins after dropping duplicates
+    if len(bins) < 2:
+        raise ValueError(
+            f"Cannot create bins: insufficient unique bin edges ({len(bins)}). "
+            "This may occur when data range is too small for float64 precision."
+        )
+
     labels = _format_labels(
         bins, precision, right=right, include_lowest=include_lowest
     )
```

This would provide a clear error message instead of silently returning all NaN.