# Bug Report: scipy.signal.deconvolve Round-Trip Property Violated with Small Leading Coefficient

**Target**: `scipy.signal.deconvolve`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.signal.deconvolve` violates its documented round-trip property `signal = convolve(divisor, quotient) + remainder` when the divisor's leading coefficient is extremely small (near float minimum), producing incorrect results due to numerical instability.

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

**Failing input**: `signal=[1.0, 1.0], divisor=[1.1754943508222875e-38, 1.0]`

## Reproducing the Bug

```python
import numpy as np
import scipy.signal

signal = np.array([1.0, 1.0])
divisor = np.array([1.1754943508222875e-38, 1.0])

quotient, remainder = scipy.signal.deconvolve(signal, divisor)

reconstructed = scipy.signal.convolve(divisor, quotient) + remainder

print(f"Original signal:  {signal}")
print(f"Reconstructed:    {reconstructed}")
print(f"Difference:       {reconstructed - signal}")
```

Output:
```
Original signal:  [1. 1.]
Reconstructed:    [1. 0.]
Difference:       [ 0. -1.]
```

The reconstructed signal `[1., 0.]` differs from the original `[1., 1.]` by a full unit in the second element.

## Why This Is A Bug

The docstring for `deconvolve` explicitly states: "Returns the quotient and remainder such that `signal = convolve(divisor, quotient) + remainder`". This is a fundamental mathematical property that should hold for all valid inputs.

When `divisor[0]` is extremely small, the `lfilter` function (used internally by `deconvolve`) divides by `divisor[0]` to normalize the filter coefficients. This division by a number near float minimum causes catastrophic numerical overflow in the quotient, which then propagates through the remainder calculation, violating the documented property.

## Fix

The function should detect when the divisor's leading coefficient is too small for stable computation and either:

1. Normalize the divisor by removing the small leading coefficient and adjusting the algorithm, or
2. Raise an informative error indicating the divisor is ill-conditioned

A high-level fix approach:

```diff
diff --git a/scipy/signal/_signaltools.py b/scipy/signal/_signaltools.py
index 1234567..abcdefg 100644
--- a/scipy/signal/_signaltools.py
+++ b/scipy/signal/_signaltools.py
@@ -2460,6 +2460,13 @@ def deconvolve(signal, divisor):
     num = xpx.atleast_nd(xp.asarray(signal), ndim=1, xp=xp)
     den = xpx.atleast_nd(xp.asarray(divisor), ndim=1, xp=xp)
+
+    # Check for numerical stability
+    if xp.abs(den[0]) < 1e-30:
+        raise ValueError(
+            f"Divisor leading coefficient {den[0]} is too small for stable "
+            "deconvolution. Please normalize or use a different divisor."
+        )
+
     if num.ndim > 1:
         raise ValueError("signal must be 1-D.")
     if den.ndim > 1: