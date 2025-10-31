# Bug Report: numpy.polynomial.Polynomial divmod Numerical Instability

**Target**: `numpy.polynomial.Polynomial.__divmod__`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `divmod` operation produces incorrect results when the divisor has a very small (near-zero) leading coefficient, violating the fundamental division property `dividend = quotient * divisor + remainder`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import numpy.polynomial as P

@settings(max_examples=200)
@given(
    coefs1=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=5),
    coefs2=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=5)
)
def test_divmod_property(coefs1, coefs2):
    p1 = P.Polynomial(coefs1)
    p2 = P.Polynomial(coefs2)

    try:
        q, r = divmod(p1, p2)
        p_reconstructed = q * p2 + r

        test_points = np.linspace(-0.5, 0.5, 5)
        original_values = p1(test_points)
        reconstructed_values = p_reconstructed(test_points)

        if not np.any(np.isnan(reconstructed_values)) and not np.any(np.isinf(reconstructed_values)):
            np.testing.assert_allclose(original_values, reconstructed_values, rtol=1e-7, atol=1e-7)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        pass
```

**Failing input**: `coefs1=[1.0, 1.0], coefs2=[1.0, 1e-200]`

## Reproducing the Bug

```python
import numpy as np
import numpy.polynomial as P

p1 = P.Polynomial([1.0, 1.0])
p2 = P.Polynomial([1.0, 1e-200])

q, r = divmod(p1, p2)
reconstructed = q * p2 + r

print(f"Expected coefficients: {p1.coef}")
print(f"Got coefficients: {reconstructed.coef}")

x = 0.5
print(f"\nExpected p1({x}) = {p1(x)}")
print(f"Got (q*p2 + r)({x}) = {reconstructed(x)}")
```

Output:
```
Expected coefficients: [1. 1.]
Got coefficients: [0. 1.]

Expected p1(0.5) = 1.5
Got (q*p2 + r)(0.5) = 0.5
```

## Why This Is A Bug

The fundamental property of polynomial division states that for any polynomials `p1` and `p2`, if `q, r = divmod(p1, p2)`, then `p1 = q * p2 + r` must hold. This bug violates this core mathematical property, producing completely incorrect results. The constant term is lost during reconstruction, causing evaluation errors of 100% or more.

## Fix

The issue stems from numerical overflow when dividing by very small coefficients in the division algorithm. The fix should add checks for near-zero leading coefficients:

```diff
--- a/numpy/polynomial/polynomial.py
+++ b/numpy/polynomial/polynomial.py
@@ -414,6 +414,9 @@ def polydiv(c1, c2):
     if len(c2) == 1:
         scl = c2[0]
+        if np.abs(scl) < np.finfo(float).eps * 100:
+            raise ValueError(f"Divisor leading coefficient ({scl}) is too small for stable division")
         c2 = c2[:-1] / scl
```
