# Bug Report: pandas.core.reshape.tile.cut - Crash with Subnormal Float Values

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.cut()` crashes with a `ValueError` when called on arrays containing very small (subnormal) floating-point numbers. The bug is in the `_round_frac()` helper function which produces `nan` values when processing subnormal floats, causing downstream failures in `IntervalIndex` creation.

## Property-Based Test

```python
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings

@given(
    data=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100),
    bins=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=500)
def test_cut_length_preservation(data, bins):
    arr = np.array(data)
    result = pd.cut(arr, bins)
    assert len(result) == len(arr), f"Expected length {len(arr)}, got {len(result)}"
```

**Failing input**: `data=[0.0, -2.2250738585072014e-308], bins=1`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

arr = np.array([0.0, -2.2250738585072014e-308])
result = pd.cut(arr, 1)
```

**Output**:
```
ValueError: missing values must be missing in the same location both left and right sides
```

## Why This Is A Bug

`pd.cut()` is documented to work on any 1-dimensional array of numeric values. Subnormal floating-point numbers (very small numbers near the limits of float64 precision) are valid IEEE 754 floats and should be handled correctly. The function should either:
1. Successfully bin these values, or
2. Raise a clear, informative error message

Instead, it raises a confusing error about "missing values" that doesn't reflect the actual problem.

## Root Cause

The bug is in `_round_frac()` at line 624 of `tile.py`:

```python
def _round_frac(x, precision: int):
    if not np.isfinite(x) or x == 0:
        return x
    else:
        frac, whole = np.modf(x)
        if whole == 0:
            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision  # BUG HERE
        else:
            digits = precision
        return np.around(x, digits)
```

For subnormal numbers like `-2.22729893e-308`:
1. `np.log10(abs(frac))` evaluates to approximately -307
2. `digits = -int(-307) - 1 + 3 = 309`
3. `np.around(x, 309)` returns `nan` due to extreme precision request

These `nan` values propagate to `IntervalIndex.from_breaks()`, causing the confusing error message.

## Fix

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -619,10 +619,16 @@ def _round_frac(x, precision: int):
     if not np.isfinite(x) or x == 0:
         return x
     else:
         frac, whole = np.modf(x)
         if whole == 0:
-            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
+            # Guard against subnormal numbers that cause numerical issues
+            log_frac = np.log10(abs(frac))
+            if not np.isfinite(log_frac) or log_frac < -300:
+                # For very small numbers, just return the value as-is
+                return x
+            digits = -int(np.floor(log_frac)) - 1 + precision
         else:
             digits = precision
         return np.around(x, digits)
```