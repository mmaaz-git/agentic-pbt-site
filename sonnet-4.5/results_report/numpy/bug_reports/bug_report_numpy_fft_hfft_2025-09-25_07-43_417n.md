# Bug Report: numpy.fft.hfft ValueError on Single-Element Arrays

**Target**: `numpy.fft.hfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.fft.hfft` crashes with a `ValueError` when given a single-element array without explicitly specifying the `n` parameter, while other FFT functions and its inverse `ihfft` handle single-element arrays correctly.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays


@given(arrays(
    dtype=np.complex128,
    shape=st.integers(min_value=1, max_value=50),
    elements=st.complex_numbers(allow_nan=False, allow_infinity=False, max_magnitude=1e6)
))
@settings(max_examples=500)
def test_hfft_ihfft_roundtrip(a):
    result = np.fft.ihfft(np.fft.hfft(a))
    assert np.allclose(result, a)
```

**Failing input**: `a=array([0.+0.j])`

## Reproducing the Bug

```python
import numpy as np

a = np.array([1.0+0.j])
result = np.fft.hfft(a)
```

Output:
```
ValueError: Invalid number of FFT data points (0) specified.
```

## Why This Is A Bug

1. **Inconsistent behavior**: Other FFT functions (`fft`, `rfft`, `ifft`, `ihfft`) all handle single-element arrays correctly
2. **No documented restriction**: The docstring for `hfft` accepts `array_like` input without excluding length-1 arrays
3. **Invalid default calculation**: When `n` is not given, the code computes `n = 2*(m-1)` where `m` is the input length. For `m=1`, this gives `n=0`, which is invalid
4. **Breaks inverse relationship**: `ihfft` works on single-element arrays, but `hfft` doesn't, breaking their documented inverse relationship

Comparison:
```python
a = np.array([5.0+0.j])
np.fft.fft(a)    # Works: array([5.+0.j])
np.fft.ihfft(a)  # Works: array([5.-0.j])
np.fft.hfft(a)   # Crashes: ValueError
```

## Fix

The issue is in the default value calculation for `n` when not specified. The code uses `n = 2*(m-1)`, which produces 0 for single-element arrays. A reasonable fix would be to use `n = max(1, 2*(m-1))` or `n = 2*m - 1` (for odd output) as the default:

```diff
--- a/numpy/fft/_pocketfft.py
+++ b/numpy/fft/_pocketfft.py
@@ -625,7 +625,7 @@ def hfft(a, n=None, axis=-1, norm=None, out=None):
     a = asarray(a)
     if n is None:
-        n = (a.shape[axis] - 1) * 2
+        n = max(1, (a.shape[axis] - 1) * 2)
     new_norm = _swap_direction(norm)
     output = irfft(conjugate(a), n, axis, norm=new_norm, out=None)
     return output
```

Note: The exact line numbers may vary. The key change is ensuring `n` is at least 1 when computed from input length.