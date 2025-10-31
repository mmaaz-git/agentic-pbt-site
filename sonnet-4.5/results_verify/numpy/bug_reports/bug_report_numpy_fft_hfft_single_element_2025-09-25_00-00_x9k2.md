# Bug Report: numpy.fft.hfft Single-Element Array Crash

**Target**: `numpy.fft.hfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.fft.hfft` crashes with ValueError when given a single-element array, while all other FFT functions in numpy.fft handle single-element arrays correctly. This is inconsistent API behavior.

## Property-Based Test

```python
import numpy as np
import numpy.fft
from hypothesis import given, strategies as st, settings


@given(st.floats(allow_nan=False, allow_infinity=False,
                min_value=-1e6, max_value=1e6))
@settings(max_examples=100)
def test_hfft_single_element_crash(value):
    a = np.array([value])
    result = numpy.fft.hfft(a)
```

**Failing input**: `[1.0]` (or any single-element array)

## Reproducing the Bug

```python
import numpy as np

a = np.array([1.0])
result = np.fft.hfft(a)
```

Output:
```
ValueError: Invalid number of FFT data points (0) specified.
```

## Why This Is A Bug

1. **Inconsistent API**: All other FFT functions (fft, ifft, rfft, irfft, ihfft) handle single-element arrays correctly, but hfft crashes.

2. **Default parameter calculation flaw**: When input has length 1, hfft computes default n = 2*(1-1) = 0, which is invalid. The function should either:
   - Use a sensible default (e.g., n=1 or n=2)
   - Raise a clear error explaining minimum input length
   - Document the minimum input length requirement

3. **Workaround exists**: Explicitly setting n works: `hfft([1.0], n=2)` succeeds, showing the underlying algorithm supports it.

## Fix

The issue is in the default n calculation. When n is None and input length m=1, the formula `2*(m-1)` produces 0. The fix should handle this edge case:

```diff
--- a/numpy/fft/_pocketfft.py
+++ b/numpy/fft/_pocketfft.py
@@ -620,7 +620,11 @@ def hfft(a, n=None, axis=-1, norm=None, out=None):
     a = asarray(a)
     if n is None:
-        n = (a.shape[axis] - 1) * 2
+        m = a.shape[axis]
+        if m == 1:
+            n = 1
+        else:
+            n = (m - 1) * 2
     new_norm = _swap_direction(norm)
     output = irfft(conjugate(a), n, axis, norm=new_norm, out=None)
     return output
```

Alternatively, document that minimum input length is 2 and raise a descriptive error for m=1.