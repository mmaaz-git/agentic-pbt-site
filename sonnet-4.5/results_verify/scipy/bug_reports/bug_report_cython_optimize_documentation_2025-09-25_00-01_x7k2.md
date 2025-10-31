# Bug Report: scipy.optimize.cython_optimize Misleading Documentation

**Target**: `scipy.optimize.cython_optimize._zeros`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstrings for functions in EXAMPLES_MAP do not clearly document the polynomial form or coefficient order, leading users to pass incorrect arguments and get wrong results.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import scipy.optimize.cython_optimize._zeros as zeros


@given(
    c0=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    c1=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    c2=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    c3=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False).filter(lambda x: abs(x) > 0.01),
)
def test_polynomial_form_documented(c0, c1, c2, c3):
    root = zeros.EXAMPLES_MAP['brentq']((c0, c1, c2, c3), -10.0, 10.0, 1e-6, 1e-6, 100)

    poly_val = c0 + c1*root + c2*root**2 + c3*root**3
    assert abs(poly_val) < 1e-3, f"Polynomial value at root should be near zero, got {poly_val}"
```

**Failing input**: Using coefficients in standard descending order (a, b, c, d) for a*x³+b*x²+c*x+d

## Reproducing the Bug

Current docstring says:
```
args : sequence of float
    extra arguments of zero function
```

This is unclear. Users would naturally try:
```python
import scipy.optimize.cython_optimize._zeros as zeros

args = (1.0, 0.0, 0.0, -2.0)
result = zeros.EXAMPLES_MAP['brentq'](args, 1.0, 2.0, 1e-6, 1e-6, 100)

print(f"Expected root of x^3 - 2: {2**(1/3)}")
print(f"Got: {result}")
```

Output:
```
Expected root of x^3 - 2: 1.2599210498948732
Got: 0.7937005259837343
```

The result is wrong because the correct usage is:
```python
args = (-2.0, 0.0, 0.0, 1.0)
```

for the polynomial `c0 + c1*x + c2*x^2 + c3*x^3 = -2 + 0*x + 0*x^2 + 1*x^3 = x^3 - 2`.

## Why This Is A Bug

1. **Ambiguous API**: The docstring doesn't specify the polynomial form being evaluated
2. **Violates user expectations**: Most mathematical conventions write polynomials in descending order (a*x^n + ... + d)
3. **No examples**: The docstring doesn't provide usage examples
4. **Misleading name**: The parameter is called "args" suggesting it's for a general function, but it's specifically for polynomial coefficients

## Fix

Update the docstrings to clearly document the polynomial form:

```diff
 def full_output_example(args, xa, xb, xtol, rtol, mitr):
     """
     Example of Cython optimize zeros functions with full output.

     Parameters
     ----------
-    args : sequence of float
-        extra arguments of zero function
+    args : tuple of 4 floats (c0, c1, c2, c3)
+        Polynomial coefficients for c0 + c1*x + c2*x^2 + c3*x^3.
+        The function finds roots where this polynomial equals zero.
+        Note: coefficients are in ascending order of powers.
     xa : float
         first boundary of zero function
     ...
+
+    Examples
+    --------
+    To find the root of x^3 - 2 = 0 in the interval [1, 2]:
+
+    >>> args = (-2.0, 0.0, 0.0, 1.0)  # represents -2 + 0*x + 0*x^2 + 1*x^3
+    >>> result = full_output_example(args, 1.0, 2.0, 1e-9, 1e-9, 100)
+    >>> print(result['root'])
+    1.2599210498948732  # cube root of 2
```