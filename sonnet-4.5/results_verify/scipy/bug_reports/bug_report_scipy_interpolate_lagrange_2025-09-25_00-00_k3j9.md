# Bug Report: scipy.interpolate.lagrange NaN with Near-Duplicate Points

**Target**: `scipy.interpolate.lagrange`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.interpolate.lagrange` returns NaN when evaluated at data points if the input contains points that are unique floating-point values but extremely close together (e.g., `5e-324` and `0.0`).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.interpolate as interp


@settings(max_examples=300)
@given(
    st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
             min_size=5, max_size=10, unique=True)
)
def test_lagrange_interpolates_exactly(x_list):
    """
    Property: Lagrange polynomial should pass exactly through all data points
    """
    x = np.array(sorted(x_list))
    y = np.sin(x)

    poly = interp.lagrange(x, y)

    for xi, yi in zip(x, y):
        result = poly(xi)
        assert np.isclose(result, yi, rtol=1e-10, atol=1e-10), \
            f"Lagrange poly at {xi} = {result}, expected {yi}"
```

**Failing input**: `x_list = [0.0, 1.0, 2.0, 0.5, 5e-324]`

## Reproducing the Bug

```python
import numpy as np
import scipy.interpolate as interp

x = np.array([5e-324, 0.0, 0.5, 1.0, 2.0])
y = np.sin(x)

poly = interp.lagrange(x, y)

for xi, yi in zip(x, y):
    result = poly(xi)
    print(f"poly({xi:.2e}) = {result}, expected {yi:.10f}")
```

Output:
```
poly(5.00e-324) = nan, expected 0.0000000000
poly(0.00e+00) = nan, expected 0.0000000000
poly(5.00e-01) = 0.4794255386, expected 0.4794255386
poly(1.00e+00) = 0.8414709848, expected 0.8414709848
poly(2.00e+00) = 0.9092974268, expected 0.9092974268
```

## Why This Is A Bug

The Lagrange polynomial interpolation is fundamentally defined to pass exactly through all given data points. When the function accepts input points (which are technically unique floating-point values) but then returns `nan` when evaluated at those same points, this violates the mathematical definition of Lagrange interpolation.

The issue arises because `5e-324` (the smallest positive subnormal float64) and `0.0` are so close that the Lagrange basis polynomial construction involves division by extremely small differences, causing numerical overflow/underflow that produces NaN.

## Fix

The function should validate inputs to detect nearly-duplicate points and either:
1. Raise a `ValueError` with a clear error message, or
2. Warn the user about potential numerical instability

A fix would add a check in `scipy.interpolate.lagrange`:

```diff
diff --git a/scipy/interpolate/_polyint.py b/scipy/interpolate/_polyint.py
index 1234567..abcdefg 100644
--- a/scipy/interpolate/_polyint.py
+++ b/scipy/interpolate/_polyint.py
@@ -50,6 +50,14 @@ def lagrange(x, w):
     """
     from numpy import poly1d
     M = len(x)
+
+    # Check for nearly-duplicate points
+    x_sorted = np.sort(x)
+    min_sep = np.min(np.diff(x_sorted))
+    if min_sep < 1e-14 * np.max(np.abs(x_sorted)):
+        raise ValueError(
+            f"Points are too close together (min separation: {min_sep}). "
+            "This causes numerical instability in Lagrange interpolation.")
+
     p = poly1d(0.0)
     for j in range(M):
         pt = poly1d(w[j])
```