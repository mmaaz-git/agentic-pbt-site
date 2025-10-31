# Bug Report: pd.cut Crash with Mixed Denormal Floats

**Target**: `pandas.cut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.cut` crashes with a confusing ValueError when binning data that contains a mix of denormal floats and normal floats, complaining about "missing values must be missing in the same location" even though the input contains no missing values.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
import pandas as pd


@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=2, max_size=50),
    bins=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_cut_assigns_all_values(values, bins):
    assume(len(set(values)) > 1)
    result = pd.cut(values, bins=bins)
    assert len(result) == len(values)
```

**Failing input**: `values=[2.2250738585e-313, -1.0], bins=2`

## Reproducing the Bug

```python
import pandas as pd

values = [2.2250738585e-313, -1.0]
result = pd.cut(values, bins=2)
```

Output:
```
ValueError: missing values must be missing in the same location both left and right sides
```

Full traceback shows the error originates from `pandas/core/arrays/interval.py:664` in the `_validate` method of IntervalArray.

## Why This Is A Bug

The `pd.cut` function should accept any valid array-like of numeric values. The docstring specifies:

> "x : array-like
>     The input array to be binned. Must be 1-dimensional."

The input `[2.2250738585e-313, -1.0]` satisfies this requirement - it's a valid 1D array of floats. However, the function crashes with a misleading error message about "missing values" when no values are actually missing. The value 2.2250738585e-313 is a denormal (subnormal) float, which is a valid IEEE 754 floating-point number.

The error message is particularly confusing because it talks about "missing values" when the actual issue is numerical precision problems during bin edge calculation.

## Fix

The root cause is that when calculating bin edges for a range that includes denormal floats, numerical operations can produce NaN values inconsistently in the left and right arrays. The fix should:

1. Detect when bin edge calculation produces invalid values
2. Provide a clear, actionable error message
3. Or handle denormal floats more carefully using higher-precision arithmetic

A defensive fix:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -475,6 +475,13 @@ def _bins_to_cuts(
         mn, mx = (np.nanmin(x_idx), np.nanmax(x_idx))
     else:
         mn, mx = (x_idx.min(), x_idx.max())
+
+    value_range = mx - mn
+    if value_range > 0 and value_range < 1e-300:
+        raise ValueError(
+            f"Cannot create bins for extremely small value range [{mn}, {mx}]. "
+            "Consider scaling your data or using explicit bin edges.")
+
     adj = (mx - mn) * 0.001

     if right:
```