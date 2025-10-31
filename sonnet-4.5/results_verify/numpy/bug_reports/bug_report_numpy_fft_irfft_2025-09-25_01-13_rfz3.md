# Bug Report: numpy.fft.irfft Single-Element Array Crash

**Target**: `numpy.fft.irfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.fft.irfft` raises ValueError when called without the `n` parameter on single-element arrays, even though this is valid input from `rfft`.

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

**Failing input**: `x_list=[0.0]` (any single-element array fails)

## Reproducing the Bug

```python
import numpy as np

x = np.array([1.0])
rfft_result = np.fft.rfft(x)
irfft_result = np.fft.irfft(rfft_result)
```

Output:
```
ValueError: Invalid number of FFT data points (0) specified.
```

Workaround: Explicitly specify `n=1`:
```python
irfft_result = np.fft.irfft(rfft_result, n=1)
```

## Why This Is A Bug

The `irfft` function should be the inverse of `rfft`. When `irfft` is called without `n`, it uses the formula `n = 2*(m-1)` where `m` is the input length. For single-element input from `rfft`, this computes `n = 2*(1-1) = 0`, which is invalid.

This breaks the expected property that users can call `irfft(rfft(x))` without specifying `n` to recover the original signal. The error only occurs for single-element arrays.

## Fix

The `n` parameter default should handle the edge case when `m=1`. In `_pocketfft.py` around line 525, the default `n` calculation should be:

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

This ensures that for single-element rfft output, irfft defaults to `n=1` instead of `n=0`.
