# Bug Report: numpy.fft.irfft Crash on Single-Element Array

**Target**: `numpy.fft.irfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Calling `irfft` without the `n` parameter on the output of `rfft` with a single-element array causes a ValueError with a confusing error message.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

float_elements = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False
)

@given(arrays(dtype=np.float64, shape=st.integers(min_value=1, max_value=100), elements=float_elements))
def test_rfft_irfft_no_crash(a):
    rfft_result = np.fft.rfft(a)
    result = np.fft.irfft(rfft_result)
    assert len(result) > 0
```

**Failing input**: `array([0.])`

## Reproducing the Bug

```python
import numpy as np

a = np.array([0.])
rfft_result = np.fft.rfft(a)
result = np.fft.irfft(rfft_result)
```

## Why This Is A Bug

When `irfft` is called without the `n` parameter, it calculates `n = 2*(m-1)` where `m` is the length of the input. For a single-element input (which is the output of `rfft` on a 1-element array), this yields `n = 2*(1-1) = 0`, causing a crash with the error message "Invalid number of FFT data points (0) specified."

This is problematic because:
1. `rfft` accepts single-element arrays and produces valid output
2. The user never specified `n=0` - the calculation is internal
3. The error message is confusing (users didn't explicitly request 0 points)
4. There's a valid interpretation: `irfft` could return a 1-element array

## Fix

```diff
--- a/numpy/fft/_pocketfft.py
+++ b/numpy/fft/_pocketfft.py
@@ -507,7 +507,7 @@ def irfft(a, n=None, axis=-1, norm=None, out=None):
     a = asarray(a)

     if n is None:
-        n = 2 * (a.shape[axis] - 1)
+        n = max(1, 2 * (a.shape[axis] - 1))

     return _raw_fft(a, n, axis, True, False, norm, out=out)
```