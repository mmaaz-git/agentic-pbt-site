# Bug Report: numpy.polynomial.Polynomial divmod Numerical Instability

**Target**: `numpy.polynomial.polynomial.polydiv` / `numpy.polynomial.Polynomial.__divmod__`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `divmod` operation on polynomials produces incorrect results when the divisor has a very small (but non-zero) leading coefficient, causing numerical overflow and violating the fundamental property that `p1 = q*p2 + r`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy as np
from numpy.polynomial import Polynomial


small_poly_coefs = st.lists(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
    min_size=1,
    max_size=5
)


@given(small_poly_coefs, small_poly_coefs)
def test_divmod_invariant(c1, c2):
    assume(len(c2) > 0)

    p2 = Polynomial(c2)
    p2_trimmed = p2.trim(tol=1e-10)
    assume(len(p2_trimmed.coef) > 0)
    assume(abs(p2_trimmed.coef[-1]) > 1e-100)

    p1 = Polynomial(c1)

    q, r = divmod(p1, p2)
    reconstructed = q * p2 + r

    p1_trimmed = p1.trim(tol=1e-10)
    reconstructed_trimmed = reconstructed.trim(tol=1e-10)

    assert np.allclose(reconstructed_trimmed.coef, p1_trimmed.coef, rtol=1e-7, atol=1e-7)
```

**Failing input**: `p1 = Polynomial([1.0, 1.0])`, `p2 = Polynomial([1.0, 1.401298464324817e-45])`

## Reproducing the Bug

```python
import numpy as np
from numpy.polynomial import Polynomial

p1 = Polynomial([1.0, 1.0])
p2 = Polynomial([1.0, 1.401298464324817e-45])

print("Dividing (1 + x) by (1 + 1.4e-45*x):")
print(f"p1 = {p1.coef}")
print(f"p2 = {p2.coef}")

q, r = divmod(p1, p2)
reconstructed = q * p2 + r

print("\nDivmod result:")
print(f"quotient  = {q.coef}")
print(f"remainder = {r.coef}")

print("\nChecking p1 = q*p2 + r:")
print(f"Expected: {p1.coef}")
print(f"Got:      {reconstructed.coef}")
print(f"BUG: Constant term is {reconstructed.coef[0]} instead of {p1.coef[0]}")
```

Output:
```
Dividing (1 + x) by (1 + 1.4e-45*x):
p1 = [1. 1.]
p2 = [1.00000000e+00 1.40129846e-45]

Divmod result:
quotient  = [7.13664607e+44]
remainder = [-7.13664607e+44  1.00000000e+00]

Checking p1 = q*p2 + r:
Expected: [1. 1.]
Got:      [0. 1.]
BUG: Constant term is 0.0 instead of 1.0
```

## Why This Is A Bug

The `divmod` operation must satisfy the mathematical property: `dividend = quotient * divisor + remainder`. When the divisor has a very small leading coefficient (like `1.4e-45`), the division algorithm in `polydiv` (polynomial.py:414-424) divides by this tiny value, causing numerical overflow and catastrophic cancellation. This produces completely incorrect results - in the example above, the constant term is 0 instead of 1.

The code only checks for exact zero leading coefficients (line 404), but doesn't handle near-zero coefficients that cause numerical instability.

## Fix

The fix should detect when the leading coefficient is too small for stable division and either:
1. Raise an informative error, or
2. Automatically trim the near-zero coefficient and retry

Here's a proposed patch:

```diff
--- a/numpy/polynomial/polynomial.py
+++ b/numpy/polynomial/polynomial.py
@@ -401,8 +401,13 @@ def polydiv(c1, c2):
     """
     # c1, c2 are trimmed copies
     [c1, c2] = pu.as_series([c1, c2])
     if c2[-1] == 0:
-        raise ZeroDivisionError  # FIXME: add message with details to exception
+        raise ZeroDivisionError("Division by polynomial with zero leading coefficient")
+
+    # Check for near-zero leading coefficient that would cause numerical instability
+    if abs(c2[-1]) < 1e-100:
+        raise ValueError(
+            f"Division by polynomial with near-zero leading coefficient ({c2[-1]:.2e}) "
+            "would cause numerical instability. Trim the divisor first."
+        )

     # note: this is more efficient than `pu._div(polymul, c1, c2)`
```