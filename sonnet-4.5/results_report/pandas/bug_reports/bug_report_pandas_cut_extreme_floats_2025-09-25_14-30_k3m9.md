# Bug Report: pandas.core.reshape.tile cut() Extreme Float Handling

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cut()` function crashes with a ValueError when binning data containing extremely small floating-point values (near the minimum representable float64). The root cause is in the `_round_frac` helper function which calculates an excessively large number of decimal places, causing `np.around` to return NaN.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pandas as pd
import math

@given(st.lists(st.floats(-100, 100, allow_nan=False), min_size=1, max_size=50),
       st.integers(2, 10))
@settings(max_examples=100)
def test_cut_coverage(data, n_bins):
    assume(len(set(data)) > 1)
    result = pd.cut(data, bins=n_bins)
    assert len(result) == len(data)
```

**Failing input**: `data=[1.1125369292536007e-308, -1.0], n_bins=2`

## Reproducing the Bug

```python
import pandas as pd

data = [1.1125369292536007e-308, -1.0]
result = pd.cut(data, bins=2)
```

Output:
```
ValueError: missing values must be missing in the same location both left and right sides
```

## Why This Is A Bug

The `cut()` function should handle all finite floating-point values gracefully. Extremely small floats are valid inputs that should be binned correctly, not cause crashes. The function fails because:

1. `_round_frac()` at line 624 of `tile.py` calculates: `digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision`
2. For `frac = 1.1125e-308`, this produces `digits = 310`
3. `np.around(x, 310)` returns NaN due to numerical limitations
4. NaN values in bin breaks cause IntervalIndex validation to fail

## Fix

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