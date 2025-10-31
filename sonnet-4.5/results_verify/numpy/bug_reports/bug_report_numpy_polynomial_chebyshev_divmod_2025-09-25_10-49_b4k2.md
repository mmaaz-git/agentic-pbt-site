# Bug Report: numpy.polynomial.Chebyshev Divmod Property Violation

**Target**: `numpy.polynomial.Chebyshev.__divmod__`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `divmod` operation on Chebyshev polynomial instances violates the fundamental property `a == b*q + r`, introducing spurious coefficients in the reconstruction due to numerical errors in the z-series conversion algorithm.

## Property-Based Test

```python
from hypothesis import assume, given, settings, strategies as st
import numpy as np
from numpy.polynomial import Chebyshev

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
def test_chebyshev_divmod_property(coefs_a, coefs_b):
    a = Chebyshev(coefs_a)
    b = Chebyshev(coefs_b)
    assume(not np.allclose(b.coef, 0))

    q, r = divmod(a, b)
    reconstructed = b * q + r

    assert np.allclose(reconstructed.trim().coef, a.trim().coef, rtol=1e-5, atol=1e-5)
```

**Failing input**: `coefs_a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]`, `coefs_b = [50, 1]`

## Reproducing the Bug

```python
import numpy as np
from numpy.polynomial import Chebyshev

a = Chebyshev([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
b = Chebyshev([50, 1])

q, r = divmod(a, b)
reconstructed = b * q + r

print("Original a:         ", a.coef)
print("Reconstructed b*q+r:", reconstructed.trim().coef)
print("Difference:         ", reconstructed.trim().coef - a.coef)
```

**Output:**
```
Original a:          [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
Reconstructed b*q+r: [ 0. -1.  0.  0.  0.  0.  0.  0.  0.  1.]
Difference:          [ 0. -1.  0.  0.  0.  0.  0.  0.  0.  0.]
```

The reconstruction introduces a spurious `-1*T_1(x)` term!

## Why This Is A Bug

The divmod operation must satisfy `a == b*q + r`. For Chebyshev polynomials, this means that when evaluated at any point, `a(x) == (b*q)(x) + r(x)`.

The reconstruction `b*q + r` introduces coefficient errors that cause it to differ from the original polynomial `a`. This violates the fundamental divmod property and produces incorrect results for polynomial division.

## Fix

The bug is in the z-series division algorithm in `numpy/polynomial/chebyshev.py`. The function `chebdiv` (lines 749-813) converts Chebyshev series to z-series, performs division, and converts back. This double conversion accumulates numerical errors.

The issue is in `_zseries_div` (lines 210-274) and the conversion functions. When coefficients have different magnitudes, the accumulated errors in the z-series operations violate the divmod property when converted back to Chebyshev basis.

A potential fix:

```diff
--- a/numpy/polynomial/chebyshev.py
+++ b/numpy/polynomial/chebyshev.py
@@ -808,6 +808,11 @@ def chebdiv(c1, c2):
         z1 = _cseries_to_zseries(c1)
         z2 = _cseries_to_zseries(c2)
         quo, rem = _zseries_div(z1, z2)
         quo = pu.trimseq(_zseries_to_cseries(quo))
         rem = pu.trimseq(_zseries_to_cseries(rem))
+        # Correct numerical errors from z-series conversion
+        reconstructed = chebadd(chebmul(c2, quo), rem)
+        error = chebsub(c1, reconstructed)
+        if np.max(np.abs(error)) > 1e-14:
+            rem = chebadd(rem, error)
         return quo, rem
```

Alternatively, use the generic `pu._div(chebmul, c1, c2)` which avoids the z-series conversion issues.