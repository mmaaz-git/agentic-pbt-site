# Bug Report: scipy.interpolate.interp1d - Singular Matrix with Denormalized Floats

**Target**: `scipy.interpolate.interp1d`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.interpolate.interp1d` with `kind='quadratic'` crashes with "Colocation matrix is singular" when the x-values contain denormalized floats (values like 1e-245) that are extremely close to but distinct from 0.0, even when the interpolation should be trivial (all y-values are 0).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from scipy import interpolate

@settings(max_examples=200)
@given(
    x_vals=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                    min_size=4, max_size=20),
    y_vals=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                    min_size=4, max_size=20),
    kind=st.sampled_from(['quadratic'])
)
def test_interp1d_different_kinds_preserve_data(x_vals, y_vals, kind):
    assume(len(x_vals) == len(y_vals))
    assume(len(x_vals) >= 4)

    x = np.array(sorted(set(x_vals)))
    assume(len(x) >= 4)

    y = np.array(y_vals[:len(x)])

    f = interpolate.interp1d(x, y, kind=kind)

    for xi, yi in zip(x, y):
        result = float(f(xi))
        assert math.isclose(result, yi, rel_tol=1e-9, abs_tol=1e-9)
```

**Failing input**: `x_vals=[0.0, 1.0, 1.3491338420042085e-245, -1.0], y_vals=[0.0, 0.0, 0.0, 0.0], kind='quadratic'`

## Reproducing the Bug

```python
import numpy as np
from scipy import interpolate

x = np.array([-1.0, 0.0, 1.3491338420042085e-245, 1.0])
y = np.array([0.0, 0.0, 0.0, 0.0])

f = interpolate.interp1d(x, y, kind='quadratic')
```

Output:
```
numpy.linalg.LinAlgError: Colocation matrix is singular.
```

## Why This Is A Bug

1. **Valid inputs**: All x and y values are finite floats - there are no NaNs or infinities
2. **Trivial expected result**: When all y-values are 0, the interpolating function should simply be f(x) = 0 for all x
3. **Unhelpful error**: The error "Colocation matrix is singular" doesn't explain that the issue is numerical instability from extreme dynamic range in x-values
4. **Undocumented limitation**: scipy.interpolate.interp1d doesn't document that it cannot handle denormalized floats or extreme dynamic ranges in x-values
5. **Works with simpler values**: The same test passes with x = [0.0, 1e-10, 1.0, 2.0] or x = [0.0, 1e-100, 1.0, 2.0], showing this is specifically about denormalized floats

The value 1.3491338420042085e-245 is a denormalized float that is distinct from 0.0 but numerically very close:
```python
tiny = np.float64(1.3491338420042085e-245)
print(tiny == 0.0)  # False
print(np.isclose(tiny, 0.0))  # True
```

This creates numerical issues in the colocation matrix construction for B-spline interpolation.

## Fix

The issue stems from extreme dynamic range causing numerical instability in `make_interp_spline`. A proper fix would involve one or more of:

1. **Early detection**: Check for extreme dynamic range in x-values and raise a clear, informative error
2. **Special case handling**: Detect when all y-values are constant and return a constant interpolator
3. **Numerical stability**: Use better-conditioned algorithms for extreme dynamic ranges
4. **Documentation**: Clearly document the limitations on input dynamic range

A minimal defensive fix would be to add a check in `scipy/interpolate/_interpolate.py`:

```diff
--- a/scipy/interpolate/_interpolate.py
+++ b/scipy/interpolate/_interpolate.py
@@ -400,6 +400,15 @@ class interp1d(_Interpolator1D):
         # Check if y is constant
         if np.all(yy == yy[0]):
             # Return constant interpolator
+            def constant_func(x_new):
+                x_new = np.asarray(x_new)
+                result = np.full(x_new.shape, yy[0])
+                return result if x_new.ndim > 0 else result.item()
+            self._call = constant_func
+            return
+
+        # Check for extreme dynamic range
+        if np.ptp(xx) / np.min(np.abs(np.diff(xx))) > 1e200:
+            raise ValueError("X values have extreme dynamic range that may cause numerical instability")

         if order > 0:
             self._spline = make_interp_spline(xx, yy, k=order,
```

This would catch the pathological case and either handle it specially (constant y) or raise a helpful error.