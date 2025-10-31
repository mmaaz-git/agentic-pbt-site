# Bug Report: pandas.qcut Crashes with duplicates='drop' on Tiny Float Values

**Target**: `pandas.core.reshape.tile.qcut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.qcut()` with `duplicates='drop'` crashes with a confusing error message when the input contains many duplicate values and one extremely small float value (near the denormalized float range).

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings, assume

@given(
    arr=st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=20,
        max_size=100
    ),
    n_bins=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_qcut_approximately_equal_sized_bins(arr, n_bins):
    assume(len(set(arr)) >= n_bins)
    result = pd.qcut(arr, q=n_bins, duplicates='drop')
    value_counts = result.value_counts()
    if len(value_counts) > 1:
        max_bin_size = value_counts.max()
        min_bin_size = value_counts.min()
        assert max_bin_size - min_bin_size <= 2
```

**Failing input**: `arr=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5e-324], n_bins=2`

## Reproducing the Bug

```python
import pandas as pd

arr = [0.0] * 19 + [5e-324]

try:
    result = pd.qcut(arr, q=2, duplicates='drop')
    print(f"Success: {result}")
except ValueError as e:
    print(f"Error with duplicates='drop': {e}")

try:
    result = pd.qcut(arr, q=2)
except ValueError as e:
    print(f"Error without duplicates='drop': {e}")
```

Output:
```
Error with duplicates='drop': missing values must be missing in the same location both left and right sides
Error without duplicates='drop': Bin edges must be unique: Index([0.0, 0.0, 5e-324], dtype='float64').
You can drop duplicate edges by setting the 'duplicates' kwarg
```

## Why This Is A Bug

Without `duplicates='drop'`, qcut suggests using it. But when you do use `duplicates='drop'`, it crashes with a confusing error about missing values. This creates a catch-22 situation for users.

The root cause is the same as the `pd.cut()` bug with tiny floats: `_round_frac()` at line 615 of `tile.py` calculates extremely large `digits` values for tiny floats (e.g., 323 for `5e-324`). When `np.around()` is called with such large digit values, it returns NaN. This causes one bin edge to become NaN while the other remains 0.0, leading to `IntervalIndex.from_breaks()` raising the error about inconsistent missing values.

## Fix

The same fix as for the `pd.cut()` bug applies here:

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