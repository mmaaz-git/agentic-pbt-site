# Bug Report: numpy.fft.hfft Single-Element Array Crash

**Target**: `numpy.fft.hfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.fft.hfft` crashes with a `ValueError` when given a single-element complex array due to an incorrect default `n` calculation that produces `n=0`.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings
from hypothesis.extra import numpy as npst


@given(npst.arrays(
    dtype=np.complex128,
    shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100)
))
@settings(max_examples=500)
def test_hfft_accepts_single_element_arrays(arr):
    np.fft.hfft(arr)
```

**Failing input**: `array([0.+0.j])`

## Reproducing the Bug

```python
import numpy as np

arr = np.array([0.+0.j])
np.fft.hfft(arr)
```

Output:
```
ValueError: Invalid number of FFT data points (0) specified.
```

The bug breaks the documented inverse relationship between `hfft` and `ihfft`:

```python
import numpy as np

real_input = np.array([5.0])
ihfft_output = np.fft.ihfft(real_input)

np.fft.hfft(ihfft_output)
```

This crashes even though `hfft` is documented as the inverse of `ihfft`.

## Why This Is A Bug

1. The function documentation states that `hfft`/`ihfft` are inverse operations
2. `ihfft` can produce single-element complex arrays from single-element real arrays
3. `hfft` should accept these as valid input but instead crashes
4. The default `n` calculation uses `n = (m-1) * 2` where `m` is the input length, producing `n=0` for `m=1`
5. The FFT algorithm requires `n >= 1`, causing the crash

## Fix

```diff
--- a/numpy/fft/_pocketfft.py
+++ b/numpy/fft/_pocketfft.py
@@ -625,7 +625,7 @@ def hfft(a, n=None, axis=-1, norm=None, out=None):
     """
     a = asarray(a)
     if n is None:
-        n = (a.shape[axis] - 1) * 2
+        n = max(1, (a.shape[axis] - 1) * 2)
     new_norm = _swap_direction(norm)
     output = irfft(conjugate(a), n, axis, norm=new_norm, out=None)
     return output
```

Alternatively, the default could use the odd-length formula `2*m - 1` for single-element inputs to produce a length-1 output.