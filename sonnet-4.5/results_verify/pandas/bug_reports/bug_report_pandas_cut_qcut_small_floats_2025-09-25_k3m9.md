# Bug Report: pandas.cut and pandas.qcut fail with very small floating-point numbers

**Target**: `pandas.core.reshape.tile.cut` and `pandas.core.reshape.tile.qcut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `pd.cut()` and `pd.qcut()` functions crash with a `ValueError` when the input data contains very small floating-point numbers (e.g., values close to the minimum representable float like 2.2e-313). The crash occurs in the `_round_frac` helper function, which produces NaN values when attempting to round very small numbers, leading to an invalid IntervalArray.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd

@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                    min_size=5, max_size=100),
    bins=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_cut_result_length(values, bins):
    series = pd.Series(values)
    result = pd.cut(series, bins=bins)
    assert len(result) == len(series)
```

**Failing input**: `values=[0.0, 0.0, 0.0, 2.2250738585e-313, -1.0], bins=2`

## Reproducing the Bug

```python
import pandas as pd

values = [0.0, 0.0, 0.0, 2.2250738585e-313, -1.0]
series = pd.Series(values)
result = pd.cut(series, bins=2)
```

**Output:**
```
RuntimeWarning: invalid value encountered in divide
ValueError: missing values must be missing in the same location both left and right sides
```

The same issue occurs with `qcut`:

```python
import pandas as pd

values = [0.0, 0.0, 0.0, 1.0, 2.2250738585072014e-308]
series = pd.Series(values)
result = pd.qcut(series, q=3, duplicates='drop')
```

## Why This Is A Bug

According to the pandas documentation, `pd.cut()` should "bin values into discrete intervals" and return "an array-like object representing the respective bin for each value of x." The function is expected to handle all valid floating-point inputs, but it crashes when the data contains very small floating-point numbers.

The root cause is in the `_round_frac` function (line 615-627 in tile.py):

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

When `x` is a very small number like `2.2e-313`:
1. `np.modf(2.2e-313)` returns `(2.2e-313, 0.0)`
2. Since `whole == 0`, it calculates `digits = -int(np.floor(np.log10(2.2e-313))) - 1 + precision`
3. This evaluates to approximately 314 digits
4. `np.around(2.2e-313, 314)` produces NaN due to numerical overflow
5. The NaN values in bin edges cause IntervalArray validation to fail

## Fix

The issue can be fixed by adding a check in `_round_frac` to handle very small numbers that would produce an excessively large number of digits:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -619,6 +619,10 @@ def _round_frac(x, precision: int):
     if not np.isfinite(x) or x == 0:
         return x
     else:
         frac, whole = np.modf(x)
         if whole == 0:
             digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
+            # Cap digits to prevent numerical overflow in np.around
+            # Using 15 as a reasonable max for float64 precision
+            if digits > 15:
+                return x
         else:
             digits = precision
         return np.around(x, digits)
```

Alternatively, the function could catch the overflow and return the original value:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -619,9 +619,14 @@ def _round_frac(x, precision: int):
     if not np.isfinite(x) or x == 0:
         return x
     else:
         frac, whole = np.modf(x)
         if whole == 0:
             digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
         else:
             digits = precision
-        return np.around(x, digits)
+        result = np.around(x, digits)
+        # If rounding produces NaN/inf, return original value
+        if not np.isfinite(result):
+            return x
+        return result
```