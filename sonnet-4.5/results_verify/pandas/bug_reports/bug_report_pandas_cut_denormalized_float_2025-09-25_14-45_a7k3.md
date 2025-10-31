# Bug Report: pandas.cut fails with denormalized floats

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.cut` crashes with a `ValueError` when binning data containing denormalized floating-point numbers, due to numerical instability in bin edge calculation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import pandas as pd


@given(
    data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
        min_size=5,
        max_size=50,
    ),
    n_bins=st.integers(2, 10),
)
@settings(max_examples=200)
def test_cut_preserves_length(data, n_bins):
    """
    Property: cut should preserve the length of the input array.
    Evidence: cut bins values but doesn't add or remove elements.
    """
    assume(len(set(data)) >= 2)

    result = pd.cut(data, bins=n_bins)
    assert len(result) == len(data)
```

**Failing input**: `data = [0.0, 0.0, 0.0, 0.0, -2.225073858507e-311], n_bins = 2`

## Reproducing the Bug

```python
import pandas as pd

data = [0.0, 0.0, 0.0, 0.0, -2.225073858507e-311]
result = pd.cut(data, bins=2)
```

**Error**:
```
ValueError: missing values must be missing in the same location both left and right sides
```

## Why This Is A Bug

1. **Valid Input**: The input is a valid 1D array of floats. Denormalized floats (very small numbers near zero) are legitimate IEEE 754 floating-point values.

2. **No Documented Restrictions**: The `cut` documentation states the input should be "array-like" and "1-dimensional" but doesn't restrict value ranges.

3. **Unhelpful Error**: The error message "missing values must be missing in the same location both left and right sides" is cryptic and doesn't help users understand the issue.

4. **Root Cause**: When the data range is extremely small (denormalized floats), the bin edge calculation produces numerical inconsistencies, likely due to underflow/overflow in the arithmetic operations.

5. **Real-World Impact**: Users working with scientific data, normalized datasets, or numerical simulations might encounter denormalized floats and expect `cut` to handle them gracefully.

## Fix

The issue occurs in the bin edge calculation when the range is extremely small. The fix should either:

1. **Option A**: Add numerical stability checks to handle very small ranges gracefully
2. **Option B**: Detect and raise a clear error message for unsupported value ranges

Option A is preferred:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -440,6 +440,13 @@ def _bins_to_cuts(
     if is_scalar(bins):
         # Compute bin edges from data range
         mn, mx = x.min(), x.max()
+
+        # Check for numerical stability issues with very small ranges
+        rng = mx - mn
+        if rng < 1e-300:  # Near denormalized float range
+            raise ValueError(
+                f"Cannot create bins for data with very small range ({rng}). "
+                "Consider scaling your data or using explicit bin edges."
+            )

         if mn == mx:  # adjust end points before binning
             mn -= 0.001 * abs(mn) if mn != 0 else 0.001
```

Alternatively, improve numerical stability in the bin calculation itself by using different arithmetic or scaling the data temporarily during binning.