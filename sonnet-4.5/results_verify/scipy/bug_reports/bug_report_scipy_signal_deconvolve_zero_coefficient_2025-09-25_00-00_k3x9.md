# Bug Report: scipy.signal.deconvolve Crashes with Zero Leading Coefficient

**Target**: `scipy.signal.deconvolve`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.signal.deconvolve` crashes with a ValueError when the divisor's leading coefficient is zero, despite this being a mathematically valid input for polynomial division.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy as np
import scipy.signal


@given(
    signal=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=50),
    divisor=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=30),
)
@settings(max_examples=500)
def test_deconvolve_round_trip(signal, divisor):
    assume(len(divisor) <= len(signal))
    assume(any(abs(d) > 1e-6 for d in divisor))

    signal_arr = np.array(signal)
    divisor_arr = np.array(divisor)

    quotient, remainder = scipy.signal.deconvolve(signal_arr, divisor_arr)

    reconstructed = scipy.signal.convolve(divisor_arr, quotient) + remainder

    np.testing.assert_allclose(reconstructed, signal_arr, rtol=1e-10, atol=1e-10)
```

**Failing input**: `signal=[0.0, 0.0], divisor=[0.0, 1.0]`

## Reproducing the Bug

```python
import numpy as np
import scipy.signal

signal = np.array([1.0, 2.0])
divisor = np.array([0.0, 1.0])

quotient, remainder = scipy.signal.deconvolve(signal, divisor)
```

Output:
```
ValueError: BUG: filter coefficient a[0] == 0 not supported yet
```

## Why This Is A Bug

The docstring for `deconvolve` states it "returns the quotient and remainder such that `signal = convolve(divisor, quotient) + remainder`". It does not specify that `divisor[0]` must be non-zero. Polynomial division with a leading zero coefficient is mathematically valid (it just means the polynomial has degree less than its array length suggests). The function should either:

1. Handle this case correctly by normalizing the divisor, or
2. Raise a more informative error message (not labeled "BUG"), or
3. Document this precondition in the docstring

The error message itself labels this as a "BUG", suggesting the developers know this is a limitation that should be fixed.

## Fix

The function should normalize the divisor by removing leading zeros before calling `lfilter`:

```diff
diff --git a/scipy/signal/_signaltools.py b/scipy/signal/_signaltools.py
index 1234567..abcdefg 100644
--- a/scipy/signal/_signaltools.py
+++ b/scipy/signal/_signaltools.py
@@ -2460,6 +2460,11 @@ def deconvolve(signal, divisor):
     num = xpx.atleast_nd(xp.asarray(signal), ndim=1, xp=xp)
     den = xpx.atleast_nd(xp.asarray(divisor), ndim=1, xp=xp)
+
+    # Remove leading zeros from divisor
+    first_nonzero = xp.argmax(xp.abs(den) > 0)
+    if first_nonzero > 0:
+        den = den[first_nonzero:]
+
     if num.ndim > 1:
         raise ValueError("signal must be 1-D.")
     if den.ndim > 1: