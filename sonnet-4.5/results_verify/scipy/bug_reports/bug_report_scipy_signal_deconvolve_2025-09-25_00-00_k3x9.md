# Bug Report: scipy.signal.deconvolve crashes when divisor[0] == 0

**Target**: `scipy.signal.deconvolve`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.signal.deconvolve` crashes with a ValueError when the divisor array has a leading zero (divisor[0] == 0), even though this limitation is not documented and the error message itself indicates it's a known bug ("BUG: filter coefficient a[0] == 0 not supported yet").

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy as np
import scipy.signal as signal

@given(
    signal_array=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
    divisor_array=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100)
)
@settings(max_examples=500)
def test_deconvolve_round_trip(signal_array, divisor_array):
    """
    Property: deconvolve docstring states:
    "Returns the quotient and remainder such that
    signal = convolve(divisor, quotient) + remainder"
    """
    signal_arr = np.array(signal_array)
    divisor_arr = np.array(divisor_array)

    assume(np.any(np.abs(divisor_arr) > 1e-10))

    quotient, remainder = signal.deconvolve(signal_arr, divisor_arr)
    reconstructed = signal.convolve(divisor_arr, quotient, mode='full')

    if len(remainder) > 0:
        remainder_padded = np.pad(remainder, (0, len(reconstructed) - len(remainder)), mode='constant')
        reconstructed = reconstructed + remainder_padded

    min_len = min(len(signal_arr), len(reconstructed))
    assert np.allclose(signal_arr[:min_len], reconstructed[:min_len], rtol=1e-5, atol=1e-5)
```

**Failing input**: `signal_array=[0.0, 0.0], divisor_array=[0.0, 1.0]`

## Reproducing the Bug

```python
import numpy as np
import scipy.signal as signal

signal_arr = np.array([0.0, 0.0])
divisor_arr = np.array([0.0, 1.0])

quotient, remainder = signal.deconvolve(signal_arr, divisor_arr)
```

Output:
```
ValueError: BUG: filter coefficient a[0] == 0 not supported yet
```

## Why This Is A Bug

1. **Undocumented limitation**: The `deconvolve` docstring makes no mention that `divisor[0]` must be non-zero
2. **Self-admitted bug**: The error message itself says "BUG" indicating this is a known implementation limitation
3. **Mathematically valid operation**: Deconvolution with a divisor having a leading zero is mathematically well-defined - this is purely an implementation limitation
4. **Implementation detail leaking**: The function internally uses `lfilter` which has this limitation, but this is an implementation detail that shouldn't affect the API

## Fix

The issue is in `/home/npc/.local/lib/python3.13/site-packages/scipy/signal/_signaltools.py` in the `deconvolve` function:

```python
def deconvolve(signal, divisor):
    # ... validation code ...

    if D > N:
        quot = []
        rem = num
    else:
        input = xp.zeros(N - D + 1, dtype=xp.float64)
        input[0] = 1
        quot = lfilter(num, den, input)  # <-- This crashes if den[0] == 0
        rem = num - convolve(den, quot, mode='full')
    return quot, rem
```

**Option 1**: Handle divisor with leading zeros by normalizing:

```diff
def deconvolve(signal, divisor):
    xp = array_namespace(signal, divisor)

    num = xpx.atleast_nd(xp.asarray(signal), ndim=1, xp=xp)
    den = xpx.atleast_nd(xp.asarray(divisor), ndim=1, xp=xp)
    if num.ndim > 1:
        raise ValueError("signal must be 1-D.")
    if den.ndim > 1:
        raise ValueError("divisor must be 1-D.")
+
+   # Handle leading zeros in divisor by removing them
+   # This shifts the quotient result accordingly
+   first_nonzero = xp.argmax(xp.abs(den) > 0)
+   if first_nonzero > 0:
+       den = den[first_nonzero:]
+       # Adjust num to account for the shift
+       if len(num) >= first_nonzero:
+           num = num[first_nonzero:]
+       else:
+           return xp.asarray([]), num
+
    N = num.shape[0]
    D = den.shape[0]
    if D > N:
        quot = []
        rem = num
    else:
        input = xp.zeros(N - D + 1, dtype=xp.float64)
        input[0] = 1
        quot = lfilter(num, den, input)
        rem = num - convolve(den, quot, mode='full')
    return quot, rem
```

**Option 2**: Document the limitation and provide a clear error message:

```diff
def deconvolve(signal, divisor):
    xp = array_namespace(signal, divisor)

    num = xpx.atleast_nd(xp.asarray(signal), ndim=1, xp=xp)
    den = xpx.atleast_nd(xp.asarray(divisor), ndim=1, xp=xp)
    if num.ndim > 1:
        raise ValueError("signal must be 1-D.")
    if den.ndim > 1:
        raise ValueError("divisor must be 1-D.")
+   if len(den) > 0 and den[0] == 0:
+       raise ValueError(
+           "divisor[0] must be non-zero. "
+           "Consider removing leading zeros from the divisor."
+       )
+
    N = num.shape[0]
    D = den.shape[0]
    # ... rest of function
```

The first option is preferred as it makes the function more robust and handles a mathematically valid case.