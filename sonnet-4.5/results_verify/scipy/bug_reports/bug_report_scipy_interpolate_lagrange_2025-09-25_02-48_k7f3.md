# Bug Report: scipy.interpolate.lagrange Fails Interpolation Property

**Target**: `scipy.interpolate.lagrange`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `lagrange` function fails to satisfy the fundamental interpolation property when x-values are widely spaced. The returned polynomial does not pass through all input points, violating its documented behavior.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from scipy.interpolate import lagrange
import math


@given(
    st.lists(
        st.tuples(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
        ),
        min_size=2,
        max_size=15
    )
)
@settings(max_examples=1000)
def test_lagrange_interpolation_property(points):
    x_vals = np.array([p[0] for p in points])
    y_vals = np.array([p[1] for p in points])

    unique_x, indices = np.unique(x_vals, return_index=True)
    assume(len(unique_x) == len(x_vals))

    sorted_indices = np.argsort(np.abs(x_vals))
    assume(np.max(np.abs(np.diff(x_vals[sorted_indices]))) > 1e-10)

    poly = lagrange(x_vals, y_vals)

    for i in range(len(x_vals)):
        result = poly(x_vals[i])
        expected = y_vals[i]
        assert math.isclose(result, expected, rel_tol=1e-6, abs_tol=1e-6), \
            f"Interpolation failed: poly({x_vals[i]}) = {result}, expected {expected}"
```

**Failing input**: `[(0.0, 32156.0), (1.0, 0.0), (260158.0, 0.0)]`

## Reproducing the Bug

```python
import numpy as np
from scipy.interpolate import lagrange

x = np.array([0.0, 1.0, 260158.0])
y = np.array([32156.0, 0.0, 0.0])

poly = lagrange(x, y)

print("Verifying interpolation property:")
for i in range(len(x)):
    result = poly(x[i])
    expected = y[i]
    error = abs(result - expected)
    print(f"poly({x[i]}) = {result}, expected {expected}, error = {error}")
```

Output:
```
poly(0.0) = 32156.0, expected 32156.0, error = 0.0
poly(1.0) = 0.0, expected 0.0, error = 0.0
poly(260158.0) = 1.0197254596278071e-06, expected 0.0, error = 1.0197254596278071e-06
```

## Why This Is A Bug

The docstring explicitly states that `lagrange` returns "the unique polynomial of lowest degree that interpolates a given set of data." The fundamental property of an interpolating polynomial is that poly(x[i]) = y[i] for all input points.

This example:
1. Uses only 3 points (well below the documented limit of "about 20 points")
2. Has no duplicate x-values
3. Produces a polynomial that fails to pass through the third point

The function produces silently incorrect results with no error or warning, despite the input being well within documented limits.

## Fix

The underlying issue is numerical instability in the Newton's divided differences implementation when x-values span vastly different scales. The recommended fix is to:

1. Add input validation to detect problematic conditioning (e.g., check the condition number of the Vandermonde matrix)
2. Raise a clear warning or error when numerical instability is likely
3. Document the preconditions more precisely (e.g., "x-values should be on similar scales")

Alternatively, the docstring should be updated to explicitly state that the interpolation property may not hold due to numerical errors, even for small numbers of points.

```diff
--- a/scipy/interpolate/_interpolate.py
+++ b/scipy/interpolate/_interpolate.py
@@ -63,6 +63,14 @@ def lagrange(x, w):

     """

+    x = np.asarray(x)
+    w = np.asarray(w)
+
+    # Check for numerical stability issues
+    x_range = np.ptp(x)
+    x_min_spacing = np.min(np.abs(np.diff(np.sort(x))))
+    if x_range / x_min_spacing > 1e6:
+        import warnings
+        warnings.warn("X-values span very different scales, which may cause numerical instability", RuntimeWarning)
+
     M = len(x)
     p = poly1d(0.0)
```