# Bug Report: scipy.interpolate.interp1d - Catastrophic Numerical Error at Data Points

**Target**: `scipy.interpolate.interp1d`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.interpolate.interp1d` with `kind='quadratic'` produces catastrophically wrong results (error magnitude ~10^227) at original data points when x-values contain denormalized floats. This violates the fundamental property of interpolation that f(x_i) = y_i.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from scipy import interpolate
import math

@settings(max_examples=200)
@given(
    x_vals=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                    min_size=4, max_size=20),
    y_vals=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                    min_size=4, max_size=20),
    kind=st.sampled_from(['quadratic'])
)
def test_interp1d_preserves_data_points(x_vals, y_vals, kind):
    assume(len(x_vals) == len(y_vals))
    assume(len(x_vals) >= 4)

    x = np.array(sorted(set(x_vals)))
    assume(len(x) >= 4)

    y = np.array(y_vals[:len(x)])

    f = interpolate.interp1d(x, y, kind=kind)

    for xi, yi in zip(x, y):
        result = float(f(xi))
        assert math.isclose(result, yi, rel_tol=1e-9, abs_tol=1e-9), \
            f"interp1d({kind}) at data point {xi} gave {result}, expected {yi}"
```

**Failing input**: `x_vals=[0.0, 1.0, 2.0, 0.5, 1.3491338420042085e-245], y_vals=[0.0, 1.0, 0.0, 0.0, 0.0], kind='quadratic'`

## Reproducing the Bug

```python
import numpy as np
from scipy import interpolate

x = np.array([0.0, 1.3491338420042085e-245, 0.5, 1.0, 2.0])
y = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

f = interpolate.interp1d(x, y, kind='quadratic')

result = f(0.5)
print(f"f(0.5) = {result}")
print(f"Expected: 0.0")
print(f"Error: {abs(result - 0.0)}")
```

Output:
```
f(0.5) = -1.1844773043065711e+227
Expected: 0.0
Error: 1.1844773043065711e+227
```

## Why This Is A Bug

1. **Violates fundamental interpolation property**: By definition, an interpolating function must satisfy f(x_i) = y_i for all data points. This property is violated catastrophically.

2. **Silent failure**: Unlike the singular matrix bug, this doesn't raise an error - it returns wildly incorrect results that could silently corrupt downstream calculations.

3. **Extreme error magnitude**: The error is ~10^227, which is far beyond any reasonable numerical precision issue. This is a complete breakdown of the numerical algorithm.

4. **Happens at a data point**: The failure occurs when evaluating at x=0.5, which is one of the original data points in the input. These evaluations should be essentially exact.

5. **Denormalized float trigger**: The bug is triggered by a denormalized float (1.3491338420042085e-245) in the x-values, similar to the singular matrix bug but with different symptoms.

## Fix

The root cause is the same as the singular matrix bug: extreme dynamic range in x-values causes numerical instability in the B-spline basis construction. The same fixes apply:

```diff
--- a/scipy/interpolate/_interpolate.py
+++ b/scipy/interpolate/_interpolate.py
@@ -390,6 +390,11 @@ class interp1d(_Interpolator1D):
         if kind in ('linear', 'nearest', 'nearest-up', 'zero', 'previous', 'next'):
             # ... existing code for simple interpolation
         else:
+            # Check for extreme dynamic range that causes numerical issues
+            x_range = np.ptp(xx)
+            x_spacing = np.min(np.abs(np.diff(xx)))
+            if x_range / x_spacing > 1e200:
+                raise ValueError("X values have extreme dynamic range that causes numerical instability. "
+                               "This often happens with denormalized floats mixed with normal values.")
+
             # Use spline interpolation
             order = {'quadratic': 2, 'cubic': 3, 'previous': 0, 'next': 0}[kind]
```

Additionally, scipy should consider normalizing x-values internally before fitting splines, then transforming back for evaluation. This is a common technique for numerical stability.