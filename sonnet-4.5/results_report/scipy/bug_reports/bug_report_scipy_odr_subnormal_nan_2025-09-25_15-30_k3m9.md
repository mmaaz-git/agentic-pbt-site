# Bug Report: scipy.odr NaN Result with Subnormal Float Initial Guess

**Target**: `scipy.odr.ODR`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When ODR.run() is called with an initial guess (`beta0`) containing subnormal floating-point numbers, the fitting algorithm produces NaN in the result parameters instead of properly converging to the correct solution.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from scipy.odr import Data, Model, ODR


def linear_func(beta, x):
    return beta[0] * x + beta[1]


@given(
    slope=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    intercept=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    n_points=st.integers(min_value=5, max_value=50)
)
@settings(max_examples=200, deadline=5000)
def test_perfect_fit_recovery(slope, intercept, n_points):
    assume(abs(slope) > 1e-6)

    x = np.linspace(0, 10, n_points)
    y_exact = slope * x + intercept

    data = Data(x, y_exact)
    model = Model(linear_func)

    initial_guess = [slope * 0.9, intercept * 0.9] if intercept != 0 else [slope * 0.9, 1.0]

    try:
        odr_obj = ODR(data, model, beta0=initial_guess)
        result = odr_obj.run()

        assert not np.any(np.isnan(result.beta)), \
            f"Result contains NaN: {result.beta}"
    except Exception as e:
        if "Iteration limit reached" in str(e) or "not full rank" in str(e):
            assume(False)
        else:
            raise
```

**Failing input**: `slope=1.0, intercept=2.225073858507e-311, n_points=5`

## Reproducing the Bug

```python
import numpy as np
from scipy.odr import Data, Model, ODR


def linear_func(beta, x):
    return beta[0] * x + beta[1]


x = np.array([0., 2.5, 5., 7.5, 10.])
y = np.array([2.225e-311, 2.5, 5.0, 7.5, 10.0])

data = Data(x, y)
model = Model(linear_func)

initial_guess = [0.9, 2.003e-311]

odr_obj = ODR(data, model, beta0=initial_guess)
result = odr_obj.run()

print(f"Result beta: {result.beta}")
print(f"Has NaN: {np.any(np.isnan(result.beta))}")


print("\nWith sanitized initial guess (replacing subnormal with 0):")
odr_obj2 = ODR(data, model, beta0=[0.9, 0.0])
result2 = odr_obj2.run()
print(f"Result beta: {result2.beta}")
print(f"Has NaN: {np.any(np.isnan(result2.beta))}")
```

Output:
```
Result beta: [            nan 2.00256647e-311]
Has NaN: True

With sanitized initial guess (replacing subnormal with 0):
Result beta: [1. 0.]
Has NaN: False
```

## Why This Is A Bug

The ODR algorithm should handle all valid floating-point inputs, including subnormal numbers. Subnormal floats (also called denormal floats) are valid IEEE 754 floating-point numbers that represent values very close to zero. While they may indicate numerical precision issues, they should not cause the algorithm to fail and return NaN.

The function succeeds when the subnormal float in the initial guess is replaced with exactly 0.0, demonstrating that the underlying problem is solvable and the issue is specifically with how ODR handles subnormal floats in the input.

This is a **High severity** bug because:
1. It causes silent data corruption (NaN results) rather than raising an exception
2. The bug can occur with legitimate floating-point arithmetic
3. Users may not notice NaN in results, leading to incorrect scientific conclusions

## Fix

The bug can be fixed by sanitizing the initial guess to replace subnormal floats with zero before passing to the FORTRAN ODRPACK routine. This can be done in the `ODR` class initialization or in the `run()` method.

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -88,6 +88,8 @@ def _conv(obj, dtype=None):
         obj = np.asarray(obj)
     else:
         obj = np.asarray(obj, dtype)
+    # Replace subnormal floats with zero to avoid numerical issues
+    obj = np.where(np.abs(obj) < np.finfo(obj.dtype).tiny, 0.0, obj)
     if obj.shape == ():
         # Scalar.
         return obj.dtype.type(obj)
```

Alternatively, the fix could detect subnormal floats and raise a warning or error:

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -1044,6 +1044,11 @@ class ODR:
         beta0 = self.beta0[:, np.newaxis]
         if self.delta0 is None:
             delta0 = None
+
+        # Check for subnormal floats in beta0
+        if np.any((beta0 != 0) & (np.abs(beta0) < np.finfo(beta0.dtype).tiny)):
+            warn("Initial guess contains subnormal floats which may cause numerical instability",
+                 OdrWarning)

         self.output = Output(_output)
```