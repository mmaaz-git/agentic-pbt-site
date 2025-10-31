# Bug Report: pandas.cut() NaN Interval Crash on Underflow Values

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.cut()` crashes with a ValueError when binning data that contains values near the floating-point underflow limit (e.g., 2.225e-308). The internal `_round_frac()` function returns NaN for such values, leading to invalid interval boundaries.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pandas as pd
import math


@given(
    x=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=10, max_size=100),
    bins=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_cut_assigns_all_values_to_bins(x, bins):
    assume(len(set(x)) > 1)
    result = pd.cut(x, bins)
    non_na_input = sum(1 for val in x if not (math.isnan(val) if isinstance(val, float) else False))
    non_na_output = result.notna().sum()
    assert non_na_input == non_na_output
```

**Failing input**: `x=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2250738585072014e-308, -1.0], bins=2`

## Reproducing the Bug

```python
import pandas as pd

x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2250738585072014e-308, -1.0]
result = pd.cut(x, bins=2)
```

**Output**:
```
ValueError: missing values must be missing in the same location both left and right sides
```

## Why This Is A Bug

The `_round_frac()` function in `pandas/core/reshape/tile.py` computes an excessively large `digits` parameter (310 in this case) when rounding extremely small numbers. This causes `np.around(x, 310)` to return NaN, which then creates invalid interval boundaries with NaN on the right side but not the left, violating IntervalArray's invariant.

Root cause in `_round_frac()` (lines 615-627):
```python
def _round_frac(x, precision: int):
    if not np.isfinite(x) or x == 0:
        return x
    else:
        frac, whole = np.modf(x)
        if whole == 0:
            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
        else:
            digits = precision
        return np.around(x, digits)
```

For `x = 2.225e-308`, this computes `digits = 310`, and `np.around(x, 310)` returns NaN.

## Fix

```diff
diff --git a/pandas/core/reshape/tile.py b/pandas/core/reshape/tile.py
index abc123..def456 100644
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -620,7 +620,9 @@ def _round_frac(x, precision: int):
     else:
         frac, whole = np.modf(x)
         if whole == 0:
             digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
+            # Cap digits to avoid np.around returning NaN
+            digits = min(digits, 15)
         else:
             digits = precision
         return np.around(x, digits)
```

Alternatively, the function could detect when the computed digits value would cause `np.around` to fail and return the original value unchanged in such cases.