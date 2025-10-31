# Bug Report: numpy.polynomial polydiv Numerical Overflow

**Target**: `numpy.polynomial.polynomial.polydiv`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `polydiv` function produces NaN and inf values when dividing polynomials where the divisor has a very small (but non-zero) leading coefficient, causing numerical overflow.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from numpy.polynomial import Polynomial


@st.composite
def polynomials_safe(draw, max_degree=8):
    deg = draw(st.integers(min_value=0, max_value=max_degree))
    coef = draw(st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=deg+1, max_size=deg+1
    ))
    return Polynomial(coef)


@given(polynomials_safe(), polynomials_safe())
@settings(max_examples=500)
def test_divmod_no_overflow(a, b):
    if np.allclose(b.coef, 0):
        return

    try:
        q, r = divmod(a, b)
        assert not np.any(np.isnan(q.coef)) and not np.any(np.isnan(r.coef))
        assert not np.any(np.isinf(q.coef)) and not np.any(np.isinf(r.coef))
    except ZeroDivisionError:
        pass
```

**Failing input**: `a=Polynomial([0., 1.])`, `b=Polynomial([1.0, 2.22507386e-311])`

## Reproducing the Bug

```python
import numpy as np
from numpy.polynomial import Polynomial

a = Polynomial([0., 1.])
b = Polynomial([1.0, 2.22507386e-311])

q, r = divmod(a, b)

print("Quotient:", q.coef)
print("Remainder:", r.coef)
print("Contains inf:", np.any(np.isinf(q.coef)) or np.any(np.isinf(r.coef)))
```

Output:
```
Quotient: [inf]
Remainder: [-inf]
Contains inf: True
```

## Why This Is A Bug

The `polydiv` function should handle polynomial division for all valid inputs. A polynomial with a small but non-zero coefficient (2.22507386e-311) is mathematically valid. The function fails because:

1. At line 416-417 in `polynomial.py`, it computes `scl = c2[-1]` then `c2 = c2[:-1] / scl`
2. When `scl` is extremely small (2.22507386e-311), dividing by it produces overflow
3. This causes subsequent operations to produce NaN and inf values

The function only checks for exactly zero (`if c2[-1] == 0:`) but doesn't handle near-zero values that cause numerical overflow.

## Fix

Add a check for numerically unstable divisors before performing division:

```diff
--- a/polynomial.py
+++ b/polynomial.py
@@ -401,8 +401,12 @@ def polydiv(c1, c2):
     """
     # c1, c2 are trimmed copies
     [c1, c2] = pu.as_series([c1, c2])
     if c2[-1] == 0:
         raise ZeroDivisionError  # FIXME: add message with details to exception
+
+    # Check for numerically unstable division
+    if np.abs(c2[-1]) < np.finfo(float).tiny * 10:
+        raise ValueError(f"Division by polynomial with leading coefficient {c2[-1]} "
+                        "is numerically unstable. Consider trimming small coefficients first.")

     # note: this is more efficient than `pu._div(polymul, c1, c2)`
     lc1 = len(c1)
```