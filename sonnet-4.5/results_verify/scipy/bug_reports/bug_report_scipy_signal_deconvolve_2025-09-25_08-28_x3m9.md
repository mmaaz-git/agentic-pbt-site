# Bug Report: scipy.signal.deconvolve ValueError with Leading Zero Coefficient

**Target**: `scipy.signal.deconvolve`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.signal.deconvolve` raises a ValueError with message "BUG: filter coefficient a[0] == 0 not supported yet" when the divisor array has a leading zero coefficient. This violates the function's documented contract and prevents deconvolution of valid polynomial inputs.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy import signal


def array_1d_strategy():
    return st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20
    ).map(np.array)


@settings(max_examples=200, deadline=5000)
@given(array_1d_strategy(), array_1d_strategy())
def test_deconvolve_property(signal_arr, divisor):
    assume(len(divisor) > 0 and len(signal_arr) >= len(divisor))
    assume(np.max(np.abs(divisor)) > 0.1)

    quotient, remainder = signal.deconvolve(signal_arr, divisor)

    reconstructed = signal.convolve(divisor, quotient, mode='full') + remainder
    trimmed_reconstructed = reconstructed[:len(signal_arr)]

    assert np.allclose(trimmed_reconstructed, signal_arr, rtol=1e-6, atol=1e-10)
```

**Failing input**: `signal_arr=array([0.0, 0.0])`, `divisor=array([0.0, 1.0])`

## Reproducing the Bug

```python
import numpy as np
from scipy import signal

signal_arr = np.array([1.0, 2.0, 3.0])
divisor = np.array([0.0, 1.0])

quotient, remainder = signal.deconvolve(signal_arr, divisor)
```

Output:
```
ValueError: BUG: filter coefficient a[0] == 0 not supported yet
```

## Why This Is A Bug

The divisor `[0.0, 1.0]` represents the valid polynomial `0*x + 1 = 1`, which should be a valid divisor. According to the function's docstring (line 2417-2418 of `_signaltools.py`):

> Returns the quotient and remainder such that `signal = convolve(divisor, quotient) + remainder`

This property should hold for all valid divisor polynomials, including those with leading zero coefficients.

The divisor `[0.0, 1.0]` is mathematically valid and represents a constant polynomial equal to 1. The function should either:
1. Handle this case correctly by normalizing the polynomial (removing leading zeros), or
2. Provide a clear, user-friendly error message explaining why this input is not supported

Instead, the function raises an internal "BUG" error message from the underlying C extension `_sigtools._linear_filter`, which suggests this is an implementation limitation rather than invalid user input.

## Fix

The function should normalize the divisor polynomial by stripping leading zeros before processing:

```diff
--- a/scipy/signal/_signaltools.py
+++ b/scipy/signal/_signaltools.py
@@ -2455,6 +2455,14 @@ def deconvolve(signal, divisor):
     xp = array_namespace(signal, divisor)

     num = xpx.atleast_nd(xp.asarray(signal), ndim=1, xp=xp)
     den = xpx.atleast_nd(xp.asarray(divisor), ndim=1, xp=xp)
+
+    # Strip leading zeros from divisor to avoid a[0] == 0 error
+    if xp.all(den == 0):
+        raise ValueError("divisor must have at least one non-zero element")
+    first_nonzero = xp.argmax(xp.abs(den) > 0)
+    if first_nonzero > 0:
+        den = den[first_nonzero:]
+
     if num.ndim > 1:
         raise ValueError("signal must be 1-D.")
     if den.ndim > 1:
```

This fix ensures that polynomials with leading zero coefficients are normalized before being passed to `lfilter`, preventing the "BUG: filter coefficient a[0] == 0" error.