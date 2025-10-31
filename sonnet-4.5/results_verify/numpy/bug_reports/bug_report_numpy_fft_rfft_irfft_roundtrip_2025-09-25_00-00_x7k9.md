# Bug Report: numpy.fft.rfft/irfft Round-Trip Fails for Odd-Length Arrays

**Target**: `numpy.fft.rfft` and `numpy.fft.irfft`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The round-trip property `irfft(rfft(a))` fails to return the original array when the input has odd length. Instead, it returns an array with one fewer element, causing data loss and breaking a fundamental invariant that users expect from inverse transform operations.

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
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100),
        elements=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False)
    )
)
@settings(max_examples=1000)
def test_rfft_irfft_roundtrip_real_input(arr):
    result = numpy.fft.irfft(numpy.fft.rfft(arr))
    np.testing.assert_allclose(result, arr, rtol=1e-10, atol=1e-10)
```

**Failing input**: Arrays with odd length, e.g., `array([1., 2., 3.])`

## Reproducing the Bug

```python
import numpy as np

arr = np.array([1., 2., 3.])
print(f'Original: {arr}, shape: {arr.shape}')

fft_result = np.fft.rfft(arr)
print(f'After rfft: {fft_result}, shape: {fft_result.shape}')

irfft_result = np.fft.irfft(fft_result)
print(f'After irfft: {irfft_result}, shape: {irfft_result.shape}')

print(f'\nExpected shape: {arr.shape}')
print(f'Actual shape: {irfft_result.shape}')
print(f'Round-trip failed: lost one element!')
```

Output:
```
Original: [1. 2. 3.], shape: (3,)
After rfft: [ 6. +0.j        -1.5+0.8660254j], shape: (2,)
After irfft: [2.25 3.75], shape: (2,)

Expected shape: (3,)
Actual shape: (2,)
Round-trip failed: lost one element!
```

## Why This Is A Bug

The documentation for `numpy.fft.ifft` explicitly states: "ifft(fft(a)) == a to within numerical accuracy". Users reasonably expect the same round-trip property for `rfft`/`irfft`, especially since these are presented as optimized versions for real input.

The root cause is that `irfft` defaults to using the length of its input when `n` is not specified. Since `rfft` returns `n//2 + 1` complex values, both even and odd original lengths map to the same FFT result length, making it impossible for `irfft` to infer the correct original size.

**Pattern of failure:**
- Size 1: crashes (separate bug)
- Size 2: ✓ works
- Size 3: ✗ returns size 2
- Size 4: ✓ works
- Size 5: ✗ returns size 4
- Size 6: ✓ works
- Size 7: ✗ returns size 6

This violates the fundamental expectation that inverse operations should recover the original input.

## Fix

The issue requires either storing metadata about the original size or changing the API. The simplest backward-compatible fix is to store the original size in the array metadata or change how `irfft` infers the default `n` value. However, this may have backward compatibility implications.

A documentation fix would be to clearly warn users that the round-trip property only holds when `n` is explicitly specified:

```diff
diff --git a/numpy/fft/_pocketfft.py b/numpy/fft/_pocketfft.py
--- a/numpy/fft/_pocketfft.py
+++ b/numpy/fft/_pocketfft.py
@@ -490,6 +490,11 @@ def irfft(a, n=None, axis=-1, norm=None, out=None):
     """
     Computes the inverse of `rfft`.

+    .. warning::
+       To ensure round-trip property ``irfft(rfft(a)) == a`` for odd-length
+       arrays, you must explicitly pass ``n=len(a)`` to ``irfft``. Without
+       this, odd-length arrays will lose one element.
+
     Parameters
     ----------
     a : array_like
```

However, a better fix would be to make `irfft` infer `n` as `2 * (len(a) - 1)` for even results and `2 * (len(a) - 1) + 1` for odd, but this would be a breaking change. The safest approach is to add a warning in the documentation and potentially deprecate the current behavior.