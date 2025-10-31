# Bug Report: scipy.odr.ODR NaN Result with Subnormal Float Initial Guess

**Target**: `scipy.odr.ODR`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

scipy.odr.ODR.run() produces NaN values in the fitted parameters when provided with an initial guess (beta0) containing subnormal floating-point numbers, instead of converging to the correct solution or raising an appropriate error.

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

<details>

<summary>
**Failing input**: `slope=1.0, intercept=2.225073858507e-311, n_points=5`
</summary>
```
Running Hypothesis test with the failing input...
slope=1.0, intercept=2.225073858507e-311, n_points=5

Initial guess: [0.9, 2.0025664726563e-311]
Is initial guess[1] subnormal? True
Test failed: Result contains NaN: [            nan 2.00256647e-311]
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.odr import Data, Model, ODR


def linear_func(beta, x):
    return beta[0] * x + beta[1]


# Create data with very small intercept (subnormal float)
x = np.array([0., 2.5, 5., 7.5, 10.])
y = np.array([2.225e-311, 2.5, 5.0, 7.5, 10.0])

data = Data(x, y)
model = Model(linear_func)

# Initial guess with subnormal float
initial_guess = [0.9, 2.003e-311]

print("Running ODR with subnormal float in initial guess...")
print(f"Initial guess: {initial_guess}")
print(f"Is initial guess[1] subnormal? {abs(initial_guess[1]) < np.finfo(np.float64).tiny and initial_guess[1] != 0}")

odr_obj = ODR(data, model, beta0=initial_guess)
result = odr_obj.run()

print(f"\nResult beta: {result.beta}")
print(f"Has NaN: {np.any(np.isnan(result.beta))}")

print("\n" + "="*50)
print("With sanitized initial guess (replacing subnormal with 0):")
initial_guess_sanitized = [0.9, 0.0]
print(f"Initial guess: {initial_guess_sanitized}")

odr_obj2 = ODR(data, model, beta0=initial_guess_sanitized)
result2 = odr_obj2.run()
print(f"\nResult beta: {result2.beta}")
print(f"Has NaN: {np.any(np.isnan(result2.beta))}")
```

<details>

<summary>
NaN appears in result when using subnormal float in initial guess
</summary>
```
Running ODR with subnormal float in initial guess...
Initial guess: [0.9, 2.003e-311]
Is initial guess[1] subnormal? True

Result beta: [       nan 2.003e-311]
Has NaN: True

==================================================
With sanitized initial guess (replacing subnormal with 0):
Initial guess: [0.9, 0.0]

Result beta: [ 1.00000000e+00 -5.88999225e-29]
Has NaN: False
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **Subnormal floats are valid IEEE 754 floating-point numbers**: Subnormal (denormal) floats like 2.003e-311 are legitimate values between zero and the smallest normal float (2.225e-308 for float64). They are designed to provide gradual underflow and should be handled correctly by numerical algorithms.

2. **Silent data corruption**: The function produces NaN values without raising any warning or error, leading to silent corruption of results. Users may not immediately notice NaN values in their output, potentially leading to incorrect scientific or engineering conclusions.

3. **The underlying problem is solvable**: When the subnormal float in the initial guess is replaced with 0.0, the algorithm converges successfully to the correct solution. This demonstrates that the optimization problem itself is well-posed and solvable - the failure is specifically due to improper handling of subnormal floats.

4. **No documentation of this limitation**: The scipy.odr documentation does not mention any restrictions on the values that can be used in the initial guess parameter (beta0), nor does it warn about potential issues with subnormal floats.

5. **Inconsistent behavior**: The algorithm handles the same subnormal values correctly when they appear in the data (y values) but fails when they appear in the initial guess, indicating an inconsistency in how these values are processed internally.

## Relevant Context

scipy.odr is a Python wrapper around the FORTRAN-77 ODRPACK library, which implements orthogonal distance regression using a modified trust-region Levenberg-Marquardt algorithm. The bug appears to occur when the FORTRAN routine receives subnormal floats in the initial parameter vector.

The issue can be verified by checking if a value is subnormal:
```python
value = 2.003e-311
is_subnormal = abs(value) < np.finfo(np.float64).tiny and value != 0
# np.finfo(np.float64).tiny = 2.2250738585072014e-308
```

Documentation references:
- scipy.odr.ODR: https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.ODR.html
- IEEE 754 subnormal numbers: https://en.wikipedia.org/wiki/Subnormal_number

The _conv function in /home/npc/.local/lib/python3.13/site-packages/scipy/odr/_odrpack.py (lines 88-103) handles conversion of inputs but doesn't sanitize subnormal floats before passing to the FORTRAN routine.

## Proposed Fix

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -88,6 +88,11 @@ def _conv(obj, dtype=None):
     """ Convert an object to the preferred form for input to the odr routine.
     """

     if obj is None:
         return obj
     else:
         if dtype is None:
             obj = np.asarray(obj)
         else:
             obj = np.asarray(obj, dtype)
+        # Sanitize subnormal floats to avoid numerical instability in FORTRAN routine
+        if obj.dtype in [np.float32, np.float64]:
+            mask = (obj != 0) & (np.abs(obj) < np.finfo(obj.dtype).tiny)
+            obj = np.where(mask, 0.0, obj)
         if obj.shape == ():
             # Scalar.
             return obj.dtype.type(obj)
         else:
             return obj
```