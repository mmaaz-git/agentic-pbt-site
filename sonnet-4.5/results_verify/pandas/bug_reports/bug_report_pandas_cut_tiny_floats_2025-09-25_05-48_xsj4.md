# Bug Report: pandas.cut Returns All NaN for Tiny Float Values

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pd.cut()` returns all NaN values when binning arrays containing extremely small floating-point numbers (near the denormalized float range), even though all input values are valid and within the computed bin ranges.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings, assume

@given(
    arr=st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100
    ),
    n_bins=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_cut_bins_coverage(arr, n_bins):
    assume(len(set(arr)) > 1)
    result = pd.cut(arr, bins=n_bins)
    non_na_input = pd.Series(arr).notna().sum()
    non_na_result = result.notna().sum()
    assert non_na_input == non_na_result
```

**Failing input**: `arr=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2250738585072014e-308], n_bins=2`

## Reproducing the Bug

```python
import pandas as pd

arr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2250738585072014e-308]
result = pd.cut(arr, bins=2)

print(f"Input: {len(arr)} non-NA values")
print(f"Output: {result.notna().sum()} non-NA values")
print(f"Result: {result}")
```

Output:
```
Input: 10 non-NA values
Output: 0 non-NA values
Result: [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
Categories (0, interval[float64, right]): []
```

## Why This Is A Bug

The function successfully creates bins `[-2.22507386e-311, 1.11253693e-308, 2.22507386e-308]` and all input values fall within this range. However, the result incorrectly returns all NaN values with 0 categories.

The root cause is in `_round_frac()` at line 615 of `tile.py`. For extremely small floats, it calculates `digits` values like 310-313 (from `-int(np.floor(np.log10(abs(frac)))) - 1 + precision`). When `np.around()` is called with such large digit values, it returns NaN, causing all bin edges to become NaN and resulting in an empty IntervalIndex.

## Fix

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -615,6 +615,8 @@ def _round_frac(x, precision: int):
     """
     Round the fractional part of the given number
     """
+    MAX_DIGITS = 15
+
     if not np.isfinite(x) or x == 0:
         return x
     else:
@@ -622,9 +624,9 @@ def _round_frac(x, precision: int):
         if whole == 0:
             digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
         else:
             digits = precision
-        return np.around(x, digits)
+        return np.around(x, min(digits, MAX_DIGITS))


 def _infer_precision(base_precision: int, bins: Index) -> int:
```

This caps the digits parameter to prevent overflow in `np.around()`. An alternative fix would be to avoid rounding entirely for subnormal floats and use the original bin values directly.