# Bug Report: numpy.polynomial.polynomial.polydiv - Catastrophic Cancellation Violates Division Property

**Target**: `numpy.polynomial.polynomial.polydiv`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`polydiv` violates the fundamental polynomial division property `dividend = quotient * divisor + remainder` when the divisor contains very small coefficients, causing catastrophic cancellation that loses significant digits.

## Property-Based Test

```python
import numpy as np
from numpy.polynomial import polynomial as poly
from hypothesis import given, strategies as st, settings
from numpy.polynomial import polyutils


@given(
    st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=1, max_size=3),
    st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=1, max_size=3)
)
@settings(max_examples=1000)
def test_polydiv_reconstruction_property(dividend, divisor):
    from hypothesis import assume

    divisor_trimmed = polyutils.trimcoef(divisor)
    assume(len(divisor_trimmed) > 0)
    assume(not np.allclose(divisor_trimmed, 0))

    quo, rem = poly.polydiv(dividend, divisor)
    reconstructed = poly.polyadd(poly.polymul(quo, divisor), rem)

    dividend_trimmed = polyutils.trimcoef(dividend)
    reconstructed_trimmed = polyutils.trimcoef(reconstructed)

    assert len(dividend_trimmed) == len(reconstructed_trimmed)
    for i, (orig, recon) in enumerate(zip(dividend_trimmed, reconstructed_trimmed)):
        assert np.isclose(orig, recon, rtol=1e-6, atol=1e-9)
```

**Failing input**: `dividend=[1.0, 1.0], divisor=[1.0, 1e-100]`

## Reproducing the Bug

```python
import numpy as np
from numpy.polynomial import polynomial as poly

dividend = [1.0, 1.0]
divisor = [1.0, 1e-100]

quo, rem = poly.polydiv(dividend, divisor)
reconstructed = poly.polyadd(poly.polymul(quo, divisor), rem)

print(f"Dividend:      {dividend}")
print(f"Divisor:       {divisor}")
print(f"Quotient:      {quo}")
print(f"Remainder:     {rem}")
print(f"Reconstructed: {reconstructed}")

assert np.allclose(dividend, reconstructed), \
    "polydiv violates: dividend = quotient * divisor + remainder"
```

**Output:**
```
Dividend:      [1.0, 1.0]
Divisor:       [1.0, 1e-100]
Quotient:      [1.e+100]
Remainder:     [-1.e+100]
Reconstructed: [0. 1.]
AssertionError: polydiv violates: dividend = quotient * divisor + remainder
```

## Why This Is A Bug

Polynomial division **must** satisfy the property: `dividend = quotient × divisor + remainder`

This is a fundamental mathematical invariant documented in the `polydiv` docstring:

> Returns the quotient-with-remainder of two polynomials `c1` / `c2`.

When the divisor has very small high-order coefficients (e.g., `1e-100`), the algorithm produces extremely large quotient and remainder values (`~1e+100`) that cancel catastrophically during reconstruction. The floating-point addition `1e+100 + (-1e+100)` loses precision and returns `0.0` instead of the correct value `1.0`.

## Fix

The root cause is in the division algorithm in `numpy/polynomial/polynomial.py`. The current implementation divides by small coefficients without checking for potential overflow:

```python
# Line ~417 in polynomial.py
scl = c2[-1]
c2 = c2[:-1] / scl  # Division by very small scl causes overflow
```

**Recommended fix:**

1. Add a condition number check before division to detect ill-conditioned cases
2. Warn users when the divisor will cause numerical instability
3. Consider normalizing polynomials before division or using a more stable algorithm

**Example mitigation:**

```diff
--- a/numpy/polynomial/polynomial.py
+++ b/numpy/polynomial/polynomial.py
@@ -414,6 +414,10 @@ def polydiv(c1, c2):
     c2 = np.trim_zeros(np.array(c2, ndmin=1, copy=True), trim='b')

     scl = c2[-1]
+    if abs(scl) < np.finfo(float).eps * np.max(np.abs(c2)):
+        import warnings
+        warnings.warn("Division by polynomial with very small leading coefficient may lose precision",
+                     RuntimeWarning, stacklevel=2)
     c2 = c2[:-1] / scl
     c1 = c1 / scl
```

A more robust solution would normalize both polynomials before division to avoid generating extreme quotient/remainder values.