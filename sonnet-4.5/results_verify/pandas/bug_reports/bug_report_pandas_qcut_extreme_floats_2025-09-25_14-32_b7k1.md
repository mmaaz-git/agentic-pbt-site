# Bug Report: pandas.core.reshape.tile qcut() Extreme Float Handling

**Target**: `pandas.core.reshape.tile.qcut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `qcut()` function crashes with a ValueError when discretizing data containing extremely small floating-point values (near the minimum representable float64). This is the same underlying bug as in `cut()` - the `_round_frac` helper function calculates an excessively large number of decimal places, causing `np.around` to return NaN.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import pandas as pd

@given(st.lists(st.floats(0, 100, allow_nan=False), min_size=4, max_size=50))
def test_qcut_handles_extreme_floats(data):
    assume(len(set(data)) >= 4)
    result = pd.qcut(data, q=4, duplicates='drop')
    assert len(result) == len(data)
```

**Failing input**: `data=[0.0, 1.0, 2.0, 1.1125369292536007e-308]`

## Reproducing the Bug

```python
import pandas as pd

data = [0.0, 1.0, 2.0, 1.1125369292536007e-308]
result = pd.qcut(data, q=4, duplicates='drop')
```

Output:
```
ValueError: missing values must be missing in the same location both left and right sides
```

## Why This Is A Bug

The `qcut()` function should handle all finite floating-point values gracefully. Like `cut()`, it fails because the `_round_frac()` function (shared by both `cut` and `qcut`) produces NaN values when formatting bin breaks with extremely small numbers:

1. `_round_frac()` calculates too many decimal places (310) for the tiny float
2. `np.around(x, 310)` returns NaN
3. IntervalIndex creation fails due to inconsistent NaN positions in breaks

This is the same root cause as the `cut()` bug.

## Fix

Same fix as for `cut()` - clamp the digits parameter in `_round_frac`:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -619,11 +619,16 @@ def _round_frac(x, precision: int):
     if not np.isfinite(x) or x == 0:
         return x
     else:
         frac, whole = np.modf(x)
         if whole == 0:
             digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
+            # Clamp digits to prevent np.around from returning NaN
+            # numpy.around has limitations for very large decimal counts
+            if digits > 15:  # float64 has ~15-17 decimal digits of precision
+                digits = 15
         else:
             digits = precision
         return np.around(x, digits)
```