# Bug Report: numpy.fft.irfft Crashes on Single-Element Array Input

**Target**: `numpy.fft.irfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `irfft` function crashes with a ValueError when called without the `n` parameter on the output of `rfft` applied to a single-element array, due to an edge case in the default calculation that produces `n=0`.

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

if __name__ == "__main__":
    test_rfft_irfft_no_crash()
```

<details>

<summary>
**Failing input**: `array([0.])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 19, in <module>
    test_rfft_irfft_no_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 13, in test_rfft_irfft_no_crash
    def test_rfft_irfft_no_crash(a):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 15, in test_rfft_irfft_no_crash
    result = np.fft.irfft(rfft_result)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py", line 525, in irfft
    output = _raw_fft(a, n, axis, True, False, norm, out=out)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py", line 60, in _raw_fft
    raise ValueError(f"Invalid number of FFT data points ({n}) specified.")
ValueError: Invalid number of FFT data points (0) specified.
Falsifying example: test_rfft_irfft_no_crash(
    a=array([0.]),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np

a = np.array([0.])
print(f"Original array: {a}")
print(f"Original array shape: {a.shape}")

rfft_result = np.fft.rfft(a)
print(f"rfft result: {rfft_result}")
print(f"rfft result shape: {rfft_result.shape}")

print("\nAttempting irfft without n parameter...")
try:
    result = np.fft.irfft(rfft_result)
    print(f"irfft result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nAttempting irfft with n=1...")
try:
    result = np.fft.irfft(rfft_result, n=1)
    print(f"irfft result with n=1: {result}")
    print(f"Does it match original? {np.allclose(result, a)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError: Invalid number of FFT data points (0) specified
</summary>
```
Original array: [0.]
Original array shape: (1,)
rfft result: [0.+0.j]
rfft result shape: (1,)

Attempting irfft without n parameter...
Error: ValueError: Invalid number of FFT data points (0) specified.

Attempting irfft with n=1...
irfft result with n=1: [0.]
Does it match original? True
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Mathematical inconsistency**: The FFT of a single-element array is mathematically well-defined. The function `rfft` correctly accepts single-element arrays and produces valid output `[a[0]+0j]`. However, `irfft` cannot process this valid output without explicit parameters.

2. **Documented formula edge case**: The documentation states that when `n` is not given, it defaults to `2*(m-1)` where `m` is the input length. For a single-element input (`m=1`), this formula yields `2*(1-1) = 0`. The code at line 524 in `/numpy/fft/_pocketfft.py` implements exactly this: `n = (a.shape[axis] - 1) * 2`, which produces an invalid value of 0 FFT data points.

3. **API inconsistency**: Every other array size works with the default parameters. Testing shows that arrays of size 2+ work correctly, making size 1 an unexpected special case that breaks the pattern.

4. **Misleading error message**: The error "Invalid number of FFT data points (0) specified" is confusing because the user never specified 0 points - this came from the internal calculation. Users would reasonably expect the function to handle this edge case automatically.

5. **Round-trip expectation violation**: The documentation promises that `irfft(rfft(a), len(a)) == a`, but for single-element arrays, even `irfft(rfft(a))` crashes rather than attempting a sensible default.

## Relevant Context

The issue is located in `/numpy/fft/_pocketfft.py` at line 524. When `n` is None and the input has shape (1,) along the specified axis, the calculation `(a.shape[axis] - 1) * 2` evaluates to 0, which is then passed to `_raw_fft` function. The `_raw_fft` function correctly rejects n=0 as invalid at line 60.

Interestingly, when explicitly specifying `n=1`, the function works correctly and produces the expected round-trip result, showing that the underlying FFT machinery can handle single-element arrays properly.

This bug only affects the default parameter calculation logic, not the core FFT functionality.

Documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html

## Proposed Fix

```diff
--- a/numpy/fft/_pocketfft.py
+++ b/numpy/fft/_pocketfft.py
@@ -521,7 +521,7 @@ def irfft(a, n=None, axis=-1, norm=None, out=None):
     """
     a = asarray(a)
     if n is None:
-        n = (a.shape[axis] - 1) * 2
+        n = max(1, (a.shape[axis] - 1) * 2)
     output = _raw_fft(a, n, axis, True, False, norm, out=out)
     return output
```