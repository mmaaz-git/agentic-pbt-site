# Bug Report: numpy.fft.hfft ValueError on Single-Element Arrays

**Target**: `numpy.fft.hfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.fft.hfft` crashes with `ValueError: Invalid number of FFT data points (0) specified.` when given a single-element array without an explicit `n` parameter, breaking the documented inverse relationship with `ihfft` and behaving inconsistently with other FFT functions.

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


if __name__ == "__main__":
    test_hfft_ihfft_roundtrip()
```

<details>

<summary>
**Failing input**: `a=array([0.+0.j])`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 18, in <module>
  |     test_hfft_ihfft_roundtrip()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 7, in test_hfft_ihfft_roundtrip
  |     dtype=np.complex128,
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 14, in test_hfft_ihfft_roundtrip
    |     assert np.allclose(result, a)
    |            ~~~~~~~~~~~^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_hfft_ihfft_roundtrip(
    |     a=array([0.+1.j, 0.+1.j]),
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 13, in test_hfft_ihfft_roundtrip
    |     result = np.fft.ihfft(np.fft.hfft(a))
    |                           ~~~~~~~~~~~^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py", line 628, in hfft
    |     output = irfft(conjugate(a), n, axis, norm=new_norm, out=None)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py", line 525, in irfft
    |     output = _raw_fft(a, n, axis, True, False, norm, out=out)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py", line 60, in _raw_fft
    |     raise ValueError(f"Invalid number of FFT data points ({n}) specified.")
    | ValueError: Invalid number of FFT data points (0) specified.
    | Falsifying example: test_hfft_ihfft_roundtrip(
    |     a=array([0.+0.j]),
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np

# Reproducing the bug with a single-element array
a = np.array([1.0+0.j])
print(f"Input array: {a}")
print(f"Input shape: {a.shape}")

try:
    result = np.fft.hfft(a)
    print(f"Result: {result}")
except ValueError as e:
    print(f"Error: {e}")

# Show that it works when n is explicitly provided
print("\nWith explicit n=1:")
result_with_n = np.fft.hfft(a, n=1)
print(f"Result with n=1: {result_with_n}")

# Compare with other FFT functions
print("\nComparing with other FFT functions on single-element array:")
print(f"fft(a): {np.fft.fft(a)}")
print(f"ifft(a): {np.fft.ifft(a)}")
print(f"rfft(a.real): {np.fft.rfft(a.real)}")
print(f"ihfft(a.real): {np.fft.ihfft(a.real)}")

# Show inverse relationship failure
print("\nInverse relationship test:")
b = np.array([1.0, 2.0])
print(f"Multi-element array: {b}")
print(f"ihfft(hfft(b)): {np.fft.ihfft(np.fft.hfft(b))}")

# Single element breaks the relationship
print(f"\nSingle-element array: {a.real}")
try:
    forward = np.fft.hfft(a.real)
    inverse = np.fft.ihfft(forward)
    print(f"ihfft(hfft(a)): {inverse}")
except ValueError as e:
    print(f"hfft fails with: {e}")
    print(f"But ihfft works: {np.fft.ihfft(a.real)}")
```

<details>

<summary>
ValueError: Invalid number of FFT data points (0) specified.
</summary>
```
Input array: [1.+0.j]
Input shape: (1,)
Error: Invalid number of FFT data points (0) specified.

With explicit n=1:
Result with n=1: [1.]

Comparing with other FFT functions on single-element array:
fft(a): [1.+0.j]
ifft(a): [1.+0.j]
rfft(a.real): [1.+0.j]
ihfft(a.real): [1.-0.j]

Inverse relationship test:
Multi-element array: [1. 2.]
ihfft(hfft(b)): [1.-0.j 2.-0.j]

Single-element array: [1.]
hfft fails with: Invalid number of FFT data points (0) specified.
But ihfft works: [1.-0.j]
```
</details>

## Why This Is A Bug

1. **Violates documented behavior**: The function accepts `array_like` input per documentation, with no restriction on minimum size. Single-element arrays are valid `array_like` objects.

2. **Breaks inverse relationship**: The documentation at lines 591-592 of `_pocketfft.py` explicitly promises:
   - `ihfft(hfft(a, 2*len(a) - 2)) == a` for even-length outputs
   - `ihfft(hfft(a, 2*len(a) - 1)) == a` for odd-length outputs
   This promise fails when `hfft` crashes but `ihfft` works correctly on single-element arrays.

3. **Inconsistent API behavior**: Other FFT functions handle single-element arrays correctly:
   - `fft`, `ifft`, `rfft` all work with single-element arrays
   - `ihfft` (the inverse of `hfft`) works with single-element arrays
   - Only `hfft` and `irfft` fail due to the same flawed default calculation

4. **Invalid default calculation**: Line 626 computes `n = (a.shape[axis] - 1) * 2`. For a single-element array (length 1), this produces `n = (1 - 1) * 2 = 0`, which is rejected by `_raw_fft` at line 60 as invalid.

5. **No documented edge case**: The documentation doesn't mention this limitation or provide guidance for handling single-element arrays.

## Relevant Context

The bug stems from the default `n` calculation when not explicitly provided. The formula `2*(m-1)` is mathematically correct for the typical use case where `hfft` processes Hermitian-symmetric data to produce real output. However, it fails to account for the edge case of single-element arrays.

Similar issue exists in `irfft` (line 524) which uses the same formula, causing it to also fail on single-element arrays. This suggests a systematic oversight in handling this edge case across real FFT functions.

The workaround is simple - users can specify `n=1` explicitly - but this requires special-case handling in user code, breaking the otherwise uniform API.

Documentation references:
- `hfft` documentation: `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/fft/_pocketfft.py:530-629`
- Default n calculation: Line 626
- Error raised: Line 60 in `_raw_fft`

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

@@ -623,7 +623,7 @@ def hfft(a, n=None, axis=-1, norm=None, out=None):
     """
     a = asarray(a)
     if n is None:
-        n = (a.shape[axis] - 1) * 2
+        n = max(1, (a.shape[axis] - 1) * 2)
     new_norm = _swap_direction(norm)
     output = irfft(conjugate(a), n, axis, norm=new_norm, out=None)
     return output
```