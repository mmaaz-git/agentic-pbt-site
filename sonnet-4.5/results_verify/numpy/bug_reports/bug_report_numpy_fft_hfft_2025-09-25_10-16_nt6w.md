# Bug Report: numpy.fft.hfft Crashes on Length-1 Input Without n Parameter

**Target**: `numpy.fft.hfft`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.fft.hfft()` crashes with a confusing error message when called on length-1 arrays without specifying the `n` parameter, despite working correctly when `n=1` is explicitly provided.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays


@given(arrays(
    dtype=np.complex128,
    shape=st.integers(min_value=1, max_value=100),
    elements=st.complex_numbers(
        allow_nan=False, allow_infinity=False,
        min_magnitude=1e-10, max_magnitude=1e6
    )
))
@settings(max_examples=500)
def test_hfft_handles_all_lengths(arr):
    result = np.fft.hfft(arr)
    assert len(result) >= 0
```

**Failing input**: `array([1.+0.j])`

## Reproducing the Bug

```python
import numpy as np

arr = np.array([1.0+0.0j])

result = np.fft.hfft(arr)
```

Output:
```
ValueError: Invalid number of FFT data points (0) specified.
```

## Why This Is A Bug

1. The function works correctly when `n=1` is explicitly specified: `np.fft.hfft(arr, n=1)` returns `[1.]`
2. The default formula `n = 2*(m-1)` produces `n=0` for length-1 inputs, causing a crash
3. No documentation warns that length-1 arrays require explicit `n` parameter
4. The error message is confusing - it mentions "0 data points" rather than explaining the actual issue
5. Length-1 Hermitian arrays are mathematically valid and should be supported

## Fix

```diff
--- a/numpy/fft/_pocketfft.py
+++ b/numpy/fft/_pocketfft.py
@@ -620,7 +620,7 @@ def hfft(a, n=None, axis=-1, norm=None, out=None):
     a = asarray(a)
     if n is None:
         n = (a.shape[axis] - 1) * 2
-        if n < 0:
+        if n < 1:
-            n = 0
+            n = 1
     new_norm = _swap_direction(norm)
     output = irfft(conjugate(a), n, axis, norm=new_norm, out=None)
```