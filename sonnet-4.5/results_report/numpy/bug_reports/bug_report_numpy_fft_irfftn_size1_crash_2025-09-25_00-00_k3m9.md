# Bug Report: numpy.fft.irfftn Crashes with Size-1 Arrays

**Target**: `numpy.fft.irfftn` and `numpy.fft.irfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.fft.irfftn` (and `numpy.fft.irfft`) crash with a `ValueError` when attempting to invert the FFT of a size-1 array. This violates the round-trip property for valid input data.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra import numpy as npst
import numpy as np
import numpy.fft
from hypothesis import strategies as st

@given(
    npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=20),
        elements=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False)
    )
)
@settings(max_examples=500)
def test_rfftn_irfftn_roundtrip(arr):
    result = numpy.fft.irfftn(numpy.fft.rfftn(arr))
    np.testing.assert_allclose(result, arr, rtol=1e-10, atol=1e-10)
```

**Failing input**: `array([0.])` (any size-1 array)

## Reproducing the Bug

```python
import numpy as np

arr = np.array([5.0])
print(f'Original: {arr}, shape: {arr.shape}')

fft_result = np.fft.rfftn(arr)
print(f'After rfftn: {fft_result}, shape: {fft_result.shape}')

try:
    irfft_result = np.fft.irfftn(fft_result)
    print(f'After irfftn: {irfft_result}')
except ValueError as e:
    print(f'ERROR: {e}')
```

Output:
```
Original: [5.], shape: (1,)
After rfftn: [5.+0.j], shape: (1,)
ERROR: Invalid number of FFT data points (0) specified.
```

Also fails with multidimensional arrays:
```python
arr2d = np.array([[5.0]])
fft_result2d = np.fft.rfftn(arr2d)
irfft_result2d = np.fft.irfftn(fft_result2d)
```

Output:
```
ValueError: Invalid number of FFT data points (0) specified.
```

## Why This Is A Bug

Size-1 arrays are valid inputs to FFT operations. The forward transform (`rfftn`) successfully handles them, but the inverse transform (`irfftn`) crashes. This breaks the fundamental round-trip property and makes these functions unreliable for generic array processing where sizes might be 1.

The error occurs in `_raw_fft` at line 60 in `_pocketfft.py`, where the code raises an error if `n < 1`. The issue is that `irfft` is being called with `n=0` instead of `n=1` when processing size-1 FFT results.

## Fix

The issue is in how `irfft` calculates the default value of `n` from its input. For size-1 input to `rfftn`, the last dimension becomes size 1 after `rfft`, and then `irfft` incorrectly infers `n=0`.

The fix should ensure that when the FFT result has size 1, `irfft` uses `n=1` as the default:

```diff
diff --git a/numpy/fft/_pocketfft.py b/numpy/fft/_pocketfft.py
--- a/numpy/fft/_pocketfft.py
+++ b/numpy/fft/_pocketfft.py
@@ -510,7 +510,10 @@ def irfft(a, n=None, axis=-1, norm=None, out=None):
     a = asarray(a)
     if n is None:
-        n = (a.shape[axis] - 1) * 2
+        m = a.shape[axis]
+        if m == 1:
+            n = 1
+        else:
+            n = (m - 1) * 2
     return _raw_fft(a, n, axis, True, False, norm, out=out)
```

This ensures that size-1 FFT results are correctly inverted to size-1 arrays, maintaining the round-trip property for all valid inputs.