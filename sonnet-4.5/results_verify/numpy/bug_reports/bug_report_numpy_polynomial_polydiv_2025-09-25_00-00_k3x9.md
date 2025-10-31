# Bug Report: numpy.polynomial.polynomial.polydiv Near-Zero Trailing Coefficient

**Target**: `numpy.polynomial.polynomial.polydiv`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`polydiv` produces incorrect results when the divisor has a near-zero (but non-zero) trailing coefficient, violating the fundamental division property: `dividend = quotient × divisor + remainder`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from numpy.polynomial import polynomial as P
from numpy.polynomial import polyutils as pu


@given(
    c1=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=10).filter(lambda x: x[-1] != 0),
    c2=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=10).filter(lambda x: x[-1] != 0)
)
@settings(max_examples=500)
def test_polydiv_property(c1, c2):
    c1_arr = np.array(c1)
    c2_arr = np.array(c2)

    quo, rem = P.polydiv(c1_arr, c2_arr)
    reconstructed = P.polyadd(P.polymul(quo, c2_arr), rem)

    c1_trimmed = pu.trimseq(c1_arr)
    reconstructed_trimmed = pu.trimseq(reconstructed)

    min_len = min(len(c1_trimmed), len(reconstructed_trimmed))
    assert np.allclose(c1_trimmed[:min_len], reconstructed_trimmed[:min_len], rtol=1e-10, atol=1e-10)
```

**Failing input**: `c1=[1.0, 1.0], c2=[1.0, 8.613104064408948e-103]`

## Reproducing the Bug

```python
import numpy as np
from numpy.polynomial import polynomial as P

c1 = np.array([1.0, 1.0])
c2 = np.array([1.0, 1e-100])

quo, rem = P.polydiv(c1, c2)

reconstructed = P.polyadd(P.polymul(quo, c2), rem)

print("c1:", c1)
print("reconstructed:", reconstructed)
print("Expected: [1. 1.], Got: [0. 1.]")
```

## Why This Is A Bug

The fundamental mathematical property of polynomial division states that for any polynomials `c1` and `c2` (with `c2 != 0`), the quotient `quo` and remainder `rem` must satisfy:

```
c1 = quo × c2 + rem
```

This property is violated when `c2` has a near-zero trailing coefficient. The issue occurs because:

1. `polydiv` calls `as_series` which trims only **exact** zeros (via `trimseq`)
2. Near-zero coefficients like `1e-100` pass through untrimmed
3. The code then divides by this tiny value (`scl = c2[-1]`), causing numerical overflow
4. The huge quotient and remainder values suffer from catastrophic cancellation when reconstructing `c1`

## Fix

The fix should check for near-zero trailing coefficients using a tolerance, not exact equality:

```diff
--- a/numpy/polynomial/polynomial.py
+++ b/numpy/polynomial/polynomial.py
@@ -401,8 +401,11 @@ def polydiv(c1, c2):
     """
     # c1, c2 are trimmed copies
     [c1, c2] = pu.as_series([c1, c2])
-    if c2[-1] == 0:
-        raise ZeroDivisionError  # FIXME: add message with details to exception
+
+    # Check for zero or near-zero trailing coefficient to avoid numerical overflow
+    tol = 100 * np.finfo(c2.dtype).eps * abs(c2).max()
+    if abs(c2[-1]) <= tol:
+        raise ZeroDivisionError("Divisor has zero or near-zero leading coefficient")

     # note: this is more efficient than `pu._div(polymul, c1, c2)`
     lc1 = len(c1)
```