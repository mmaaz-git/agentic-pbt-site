# Bug Report: pandas.qcut Small Float Handling

**Target**: `pandas.core.reshape.tile.qcut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.qcut()` crashes with a ValueError when the input contains very small floating-point numbers (e.g., values around 1e-308). The crash occurs due to NaN values introduced by the `_round_frac` helper function when attempting to round extremely small numbers.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pandas as pd

@given(
    values=st.lists(
        st.floats(min_value=-100, max_value=100,
                 allow_nan=False, allow_infinity=False),
        min_size=10, max_size=100
    ),
    n_quantiles=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=100)
def test_qcut_no_crash_on_valid_input(values, n_quantiles):
    series = pd.Series(values).dropna()
    assume(len(series) >= n_quantiles)
    assume(len(series.unique()) >= n_quantiles)

    try:
        result = pd.qcut(series, q=n_quantiles)
        assert len(result) == len(series)
    except ValueError as e:
        if "Bin edges must be unique" in str(e):
            assume(False)
        raise
```

**Failing input**: `values=[0.0, 1.0, 2.0, 2.2250738585072014e-308], n_quantiles=4`

## Reproducing the Bug

```python
import pandas as pd

values = [0.0, 1.0, 2.0, 2.2250738585072014e-308]
series = pd.Series(values)

result = pd.qcut(series, q=4)
```

**Output**:
```
ValueError: missing values must be missing in the same location both left and right sides
```

## Why This Is A Bug

The input data is completely valid - all values are finite, non-NaN floats. The function should either:
1. Successfully bin the data into quantiles, or
2. Raise an informative error message explaining why the data cannot be binned

Instead, it crashes with a confusing error message about "missing values" when there are no missing values in the input.

The root cause is in `pandas.core.reshape.tile._round_frac()` at line 624:

```python
digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
```

For extremely small floats like `2.2250738585072014e-308`, this calculates `digits=310`, and `np.around(x, 310)` returns NaN, which then causes the IntervalArray validation to fail.

## Fix

The `_round_frac` function should handle very small floating point numbers more carefully. One approach is to cap the maximum number of digits used in rounding:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -621,7 +621,8 @@ def _round_frac(x, precision: int):
     else:
         frac, whole = np.modf(x)
         if whole == 0:
-            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
+            raw_digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
+            digits = min(raw_digits, 15)  # Cap at 15 decimal places
         else:
             digits = precision
         return np.around(x, digits)
```

Alternatively, the function could detect when the value is too small and return it unchanged:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -615,7 +615,10 @@ def _round_frac(x, precision: int):
 def _round_frac(x, precision: int):
     """
     Round the fractional part of the given number
     """
-    if not np.isfinite(x) or x == 0:
+    # For extremely small numbers, np.around with high precision returns NaN
+    # Treat them as effectively zero for rounding purposes
+    if not np.isfinite(x) or x == 0 or abs(x) < 1e-100:
         return x
     else:
         frac, whole = np.modf(x)
```