# Bug Report: numpy.polynomial.Polynomial.roots() Returns Invalid Roots

**Target**: `numpy.polynomial.Polynomial.roots()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a Polynomial has tiny leading coefficients (e.g., from floating-point arithmetic), `roots()` returns numerically unstable values that don't satisfy the fundamental property p(root) ≈ 0.

## Property-Based Test

```python
import numpy as np
import numpy.polynomial as np_poly
from hypothesis import assume, given, settings, strategies as st


@settings(max_examples=1000)
@given(
    coef=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
        min_size=2,
        max_size=6
    )
)
def test_polynomial_roots_are_valid(coef):
    p = np_poly.Polynomial(coef)

    assume(p.degree() >= 1)

    roots = p.roots()

    assume(not np.any(np.isnan(roots)))
    assume(not np.any(np.isinf(roots)))

    for root in roots:
        value = abs(p(root))
        assert value < 1e-6, f'p({root}) = {p(root)}, expected ~0'
```

**Failing input**: `coef=[0.0, 1.0, 3.254353641323301e-273]`

## Reproducing the Bug

```python
import numpy.polynomial as np_poly

p = np_poly.Polynomial([1.0, 1.0, 3.9968426114653685e-66])

roots = p.roots()
print(f'Computed roots: {roots}')

for root in roots:
    print(f'p({root}) = {p(root)}')
```

Output:
```
Computed roots: [-2.50197493e+65  0.00000000e+00]
p(-2.5019749267369036e+65) = 1.0
p(0.0) = 1.0
```

The polynomial's actual root is at x = -1.0 (where p(-1.0) = 0.0), but `roots()` returns invalid values where p(root) = 1.0.

## Why This Is A Bug

1. The fundamental property of roots is that p(root) = 0
2. The returned "roots" evaluate to 1.0, not 0, violating this property
3. Such tiny coefficients can arise from normal floating-point operations like subtraction
4. The docstring promises "roots of the series" but delivers numerically meaningless values

## Fix

The `roots()` method should trim tiny trailing coefficients before computing roots to avoid numerical instability:

```diff
--- a/numpy/polynomial/_polybase.py
+++ b/numpy/polynomial/_polybase.py
@@ -xxx,xxx +xxx,xxx @@ class ABCPolyBase:
     def roots(self):
         """Return the roots of the series polynomial."""
-        roots = self._roots(self.coef)
+        trimmed_coef = pu.trimcoef(self.coef, tol=1e-14)
+        roots = self._roots(trimmed_coef)
         return pu.mapdomain(roots, self.window, self.domain)
```

This ensures that polynomials with negligible high-order terms are treated according to their effective degree.