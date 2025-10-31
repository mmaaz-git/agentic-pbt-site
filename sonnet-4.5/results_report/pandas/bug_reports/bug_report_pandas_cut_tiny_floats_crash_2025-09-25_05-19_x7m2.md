# Bug Report: pandas.core.reshape.tile.cut - Crash with Mixed Sign Tiny Floats

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `pd.cut()` function crashes with a `ValueError` when given valid input containing very small positive and negative float values, rather than handling them gracefully.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings, assume


@given(
    st.lists(st.floats(min_value=-1e-300, max_value=1e-300, allow_nan=False, allow_infinity=False), min_size=2, max_size=10),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=100)
def test_cut_handles_tiny_floats_without_crash(values, n_bins):
    assume(len(set(values)) > 1)

    x = pd.Series(values)
    try:
        result = pd.cut(x, bins=n_bins)
    except ValueError as e:
        if "missing values must be missing" in str(e):
            raise AssertionError(f"cut() crashed on valid input: {values}") from e
        raise
```

**Failing input**: `values=[1.1125369292536007e-308, -6.312184867281418e-301], n_bins=2`

## Reproducing the Bug

```python
import pandas as pd

values = [1.1125369292536007e-308, -6.312184867281418e-301]
x = pd.Series(values)

print(f"Input values: {x.tolist()}")
print(f"All values are valid (non-NaN): {x.notna().all()}")

try:
    result = pd.cut(x, bins=2)
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")
```

Expected output: Valid binning result or a clear, actionable error message.

Actual output:
```
Input values: [1.1125369292536007e-308, -6.312184867281418e-301]
All values are valid (non-NaN): True
ValueError raised: missing values must be missing in the same location both left and right sides
```

## Why This Is A Bug

This is a crash on valid input data. The error message "missing values must be missing in the same location both left and right sides" is:
1. Confusing - there are no missing values in the input
2. Coming from an internal validation function (`IntervalArray._validate`) that shouldn't be reached with valid input
3. Preventing legitimate use of `pd.cut()` on small float values

The bug occurs because:
- When computing bin edges for very small float ranges, numerical precision issues cause some bin edges to become NaN
- The IntervalArray constructor then receives arrays with NaN values in inconsistent positions
- This triggers the "missing values must be missing in the same location" validation error

This is a **medium-severity** bug because:
1. It causes crashes rather than silent corruption
2. It affects a narrow range of inputs (very small floats spanning positive and negative)
3. The error message is cryptic and doesn't help users understand the problem

## Fix

The fix should ensure that:
1. Bin edge computation handles very small float ranges without producing NaN values
2. If bin edges cannot be computed reliably, raise a clear, actionable error
3. Consider using extended precision for bin edge calculations when the range is very small

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -470,6 +470,13 @@ def _bins_to_cuts(
     bins_idx = Index(bins)
     bins_idx = bins_idx.unique()

+    # Check if bin computation resulted in invalid values
+    if bins_idx.isna().any():
+        raise ValueError(
+            f"Unable to compute valid bin edges for the given data range. "
+            f"Data range: [{x_idx.min()}, {x_idx.max()}]. "
+            f"Consider using larger values or fewer bins."
+        )
+
     try:
         labels = _format_labels(
             bins, precision, right=right, include_lowest=include_lowest
```

Note: The actual root cause is likely in `_nbins_to_bins` where bin edges are computed. A more complete fix would prevent NaN values from being created in the first place.