# Bug Report: pandas.core.window Rolling Variance Returns Negative Values

**Target**: `pandas.core.window.rolling.Rolling.var`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The rolling variance calculation can return negative values due to numerical instability when processing data containing extremely small values (near machine epsilon) mixed with zeros. Mathematically, variance must be non-negative, so any negative result is incorrect.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=5, max_size=30))
def test_rolling_var_non_negative(values):
    s = pd.Series(values)
    result = s.rolling(3).var()

    for i, val in enumerate(result):
        if not np.isnan(val):
            assert val >= 0, f"At index {i}: variance {val} is negative"
```

**Failing input**: `values=[0.0, 249.5, 5e-324, 0.0, 0.0]`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

values = [0.0, 249.5, 5e-324, 0.0, 0.0]
s = pd.Series(values)

var_result = s.rolling(3).var()
print("Rolling variance:", var_result.values)
print(f"Variance at index 4: {var_result.iloc[4]}")

window = [5e-324, 0.0, 0.0]
print(f"Window: {window}")
print(f"NumPy variance (correct): {np.var(window, ddof=1)}")
```

**Output:**
```
Rolling variance: [            nan             nan  2.07500833e+04  2.07500833e+04 -3.63797881e-12]
Variance at index 4: -3.637978807091713e-12
Window: [5e-324, 0.0, 0.0]
NumPy variance (correct): 0.0
```

## Why This Is A Bug

Variance is defined as the average squared deviation from the mean and must be non-negative by definition. The pandas rolling variance implementation produces negative results (e.g., -3.6e-12) when it should return 0.0, indicating a numerical stability issue in the variance computation algorithm.

This bug causes:
1. Mathematically impossible results (negative variance)
2. Invalid downstream calculations (std = sqrt(var) produces NaN)
3. Violated fundamental statistical properties

## Fix

The issue is likely caused by using a numerically unstable one-pass variance algorithm (like Welford's algorithm with floating-point errors, or a naive sum-of-squares approach). The solution is to:

1. Use a numerically stable two-pass algorithm, or
2. Apply a post-processing step to clamp negative values to zero: `result[result < 0] = 0`
3. Consider using compensated summation (Kahan summation) for better precision

The simplest fix would be to clamp negative variances to zero:

```diff
--- a/pandas/core/window/rolling.py
+++ b/pandas/core/window/rolling.py
@@ -variance_calculation_location
+        # Ensure variance is non-negative (fix numerical instability)
+        if result < 0:
+            result = 0.0
         return result
```

Note: The exact location would need to be identified in the pandas source code, likely in the Cython variance implementation or the numba-accelerated version.