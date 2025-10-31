# Bug Report: numpy.fft irfft Family Functions Crash on Single-Element Arrays

**Target**: `numpy.fft.irfft`, `numpy.fft.irfft2`, `numpy.fft.irfftn`, `numpy.fft.hfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Multiple inverse real FFT functions crash with ValueError when the size parameter is not explicitly provided for single-element arrays. The functions `irfft`, `irfft2`, `irfftn`, and `hfft` all fail due to computing a default size of 0.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                         min_value=-1e6, max_value=1e6),
               min_size=1, max_size=100))
def test_irfft_produces_real_output(x_list):
    x = np.array(x_list, dtype=np.float64)
    rfft_result = np.fft.rfft(x)
    irfft_result = np.fft.irfft(rfft_result)
    assert irfft_result.dtype == np.float64
```

**Failing input**: Any single-element array, e.g., `x_list=[1.0]`

## Reproducing the Bug

```python
import numpy as np

print("Bug 1: irfft on single-element array")
x = np.array([1.0])
rfft_result = np.fft.rfft(x)
result = np.fft.irfft(rfft_result)

print("\nBug 2: irfft2 on single-element 2D array")
x2d = np.array([[1.0]])
rfft2_result = np.fft.rfft2(x2d)
result2 = np.fft.irfft2(rfft2_result)

print("\nBug 3: irfftn on single-element array")
rfftn_result = np.fft.rfftn(x)
result3 = np.fft.irfftn(rfftn_result)

print("\nBug 4: hfft on single-element array")
x_hermitian = np.array([1.0+0j])
result4 = np.fft.hfft(x_hermitian)
```

Output:
```
ValueError: Invalid number of FFT data points (0) specified.
```

Workaround for each function:
```python
np.fft.irfft(rfft_result, n=1)
np.fft.irfft2(rfft2_result, s=(1, 1))
np.fft.irfftn(rfftn_result, s=(1,))
np.fft.hfft(x_hermitian, n=1)
```

## Why This Is A Bug

All four functions use the default size formula `n = 2*(m-1)` where `m` is the input length. For single-element inputs, this computes `n = 2*(1-1) = 0`, which is invalid.

This violates the expected behavior that:
1. `irfft(rfft(x))` should work as the inverse operation without specifying `n`
2. Users shouldn't need to know the implementation detail of how default sizes are computed
3. Single-element arrays are valid inputs for FFT operations

The bug affects all real inverse FFT functions and breaks the natural round-trip property for the smallest possible input size.

## Fix

In `_pocketfft.py`, modify the default `n` calculation to handle the edge case:

```diff
--- a/numpy/fft/_pocketfft.py
+++ b/numpy/fft/_pocketfft.py
@@ -515,7 +515,7 @@ def irfft(a, n=None, axis=-1, norm=None, out=None):
     a = asarray(a)
     if n is None:
-        n = 2 * (a.shape[axis] - 1)
+        n = max(1, 2 * (a.shape[axis] - 1))
     return _raw_fft(a, n, axis, True, False, norm, out=out)
```

The same fix should be applied to any other functions that use this formula for default size calculation.