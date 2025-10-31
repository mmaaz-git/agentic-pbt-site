# Bug Report: numpy.polynomial Divmod Property Violation

**Target**: `numpy.polynomial.Polynomial.__divmod__` (and other polynomial types)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `divmod` operation on polynomial instances violates the fundamental mathematical property `a == b*q + r`, introducing spurious coefficients in the reconstruction. This affects `Polynomial`, `Chebyshev`, and potentially other polynomial types.

## Property-Based Test

```python
from hypothesis import assume, given, settings, strategies as st
import numpy as np
from numpy.polynomial import Polynomial

@st.composite
def polynomial_coefficients(draw):
    size = draw(st.integers(min_value=1, max_value=10))
    coefs = draw(st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=size,
        max_size=size
    ))
    coefs = [c if abs(c) >= 1e-10 else 0.0 for c in coefs]
    assume(any(c != 0 for c in coefs))
    return coefs

@given(polynomial_coefficients(), polynomial_coefficients())
@settings(max_examples=500)
def test_polynomial_divmod_property(coefs_a, coefs_b):
    a = Polynomial(coefs_a)
    b = Polynomial(coefs_b)
    assume(not np.allclose(b.coef, 0))

    q, r = divmod(a, b)
    reconstructed = b * q + r

    assert np.allclose(reconstructed.trim().coef, a.trim().coef, rtol=1e-5, atol=1e-5)
```

**Failing input**: `coefs_a = [0, 0, 0, 0, 0, 0, 0, 1]`, `coefs_b = [72, 1.75]`

## Reproducing the Bug

```python
import numpy as np
from numpy.polynomial import Polynomial

a = Polynomial([0, 0, 0, 0, 0, 0, 0, 1])
b = Polynomial([72, 1.75])

q, r = divmod(a, b)
reconstructed = b * q + r

print("Original a:         ", a.coef)
print("Reconstructed b*q+r:", reconstructed.trim().coef)
print("Difference:         ", reconstructed.trim().coef - a.coef)
```

**Output:**
```
Original a:          [0. 0. 0. 0. 0. 0. 0. 1.]
Reconstructed b*q+r: [-3.05e-05  9.54e-07  0.00e+00  4.66e-10 -1.46e-11  2.27e-13 -7.11e-15  1.00e+00]
Difference:          [-3.05e-05  9.54e-07  0.00e+00  4.66e-10 -1.46e-11  2.27e-13 -7.11e-15  0.00e+00]
```

## Why This Is A Bug

The divmod operation must satisfy `a == b*q + r` where `q` is the quotient and `r` is the remainder. This is the fundamental property of polynomial division.

In this case, dividing `x^7` by `72 + 1.75*x` should yield results where `(72 + 1.75*x)*q + r = x^7` exactly (within floating-point precision). Instead, the reconstruction introduces spurious non-zero coefficients in positions that should be zero.

This violates basic polynomial arithmetic and will produce incorrect results for any code relying on polynomial division.

## Fix

The bug is in `numpy/polynomial/polynomial.py` in the `polydiv` function (lines 414-424). The in-place modification algorithm accumulates numerical errors when coefficients have different magnitudes:

```python
while i >= 0:
    c1[i:j] -= c2 * c1[j]
    i -= 1
    j -= 1
return c1[j + 1:] / scl, pu.trimseq(c1[:j + 1])
```

The algorithm modifies `c1` in-place, and numerical errors accumulate during the subtraction operations. When coefficients like 72 and 1.75 interact with higher powers through repeated operations, floating-point errors compound.

A fix would be to use the more numerically stable `pu._div` function mentioned in line 407, or to add a verification/correction step:

```diff
--- a/numpy/polynomial/polynomial.py
+++ b/numpy/polynomial/polynomial.py
@@ -420,7 +420,10 @@ def polydiv(c1, c2):
         while i >= 0:
             c1[i:j] -= c2 * c1[j]
             i -= 1
             j -= 1
-        return c1[j + 1:] / scl, pu.trimseq(c1[:j + 1])
+        quo = c1[j + 1:] / scl
+        rem = pu.trimseq(c1[:j + 1])
+        # Correct numerical errors: adjust remainder to ensure a == b*quo + rem
+        correction = pu.as_series([c1_original])[0] - (polymul(c2_original, quo) + rem)
+        rem = rem + correction
+        return quo, pu.trimseq(rem)
```

However, this would require saving the original inputs. A better solution is to use the generic `pu._div(polymul, c1, c2)` which is more numerically stable.