# Bug Report: numpy.polynomial Polynomial.__pow__ Inconsistent Trimming

**Target**: `numpy.polynomial.Polynomial.__pow__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `__pow__` operator produces polynomials with trailing zero coefficients that differ in length from equivalent polynomials produced by repeated multiplication, causing equality checks to fail even though the polynomials are mathematically equivalent.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from numpy.polynomial import Polynomial


@settings(max_examples=500)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=6),
    st.integers(min_value=2, max_value=5)
)
def test_power_equals_repeated_multiplication(coeffs, n):
    p = Polynomial(coeffs)

    power_result = p ** n

    mult_result = Polynomial([1])
    for _ in range(n):
        mult_result = mult_result * p

    assert np.allclose(power_result.coef, mult_result.coef, atol=1e-8), \
        f"p**{n} != p*...*p (repeated {n} times)"
```

**Failing input**: `coeffs=[1.0, 3.97e-289, 0.0], n=2`

## Reproducing the Bug

```python
from numpy.polynomial import Polynomial

p = Polynomial([1, 1e-200, 0])

power = p ** 2
mult = p * p

print("p**2 coefficients:", power.coef)
print("p*p coefficients:", mult.coef)
print("Are they equal?", power == mult)
```

**Output:**
```
p**2 coefficients: [1.e+000 2.e-200 0.e+000]
p*p coefficients: [1.e+000 2.e-200]
Are they equal? False
```

## Why This Is A Bug

The power operator `**` should behave identically to repeated multiplication for mathematical consistency. When a polynomial with trailing zero coefficients is raised to a power `n >= 2`, the result contains `n * (len(original_coef) - 1)` trailing zeros, while repeated multiplication properly trims trailing zeros. This causes:

1. `p**n == p*p*...*p` to return `False` even when mathematically equivalent
2. `has_samecoef()` to fail on mathematically identical polynomials
3. Inconsistent coefficient array lengths across equivalent operations

The `__eq__` method checks `self.coef.shape == other.coef.shape`, so even though the polynomials evaluate identically at all points, they are considered unequal.

## Fix

The `__pow__` method should trim the result before returning, consistent with multiplication:

```diff
--- a/numpy/polynomial/_polybase.py
+++ b/numpy/polynomial/_polybase.py
@@ -xxx,xx +xxx,xx @@ class ABCPolyBase:
     def __pow__(self, other):
         coef = self._pow(self.coef, other, maxpower=self.maxpower)
         res = self.__class__(coef, self.domain, self.window, self.symbol)
-        return res
+        return res.trim()
```

This ensures `p**n` produces the same coefficient array as repeated multiplication, maintaining mathematical consistency.