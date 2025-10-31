# Bug Report: numpy.fft irfft Family Functions Crash on Single-Element Arrays

**Target**: `numpy.fft.irfft`, `numpy.fft.irfft2`, `numpy.fft.irfftn`, `numpy.fft.hfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The inverse real FFT functions (`irfft`, `irfft2`, `irfftn`, `hfft`) crash with ValueError when processing single-element arrays without an explicit size parameter, computing an invalid default size of 0.

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

if __name__ == "__main__":
    test_irfft_produces_real_output()
```

<details>

<summary>
**Failing input**: `x_list=[0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 14, in <module>
    test_irfft_produces_real_output()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 5, in test_irfft_produces_real_output
    min_value=-1e6, max_value=1e6),

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 10, in test_irfft_produces_real_output
    irfft_result = np.fft.irfft(rfft_result)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py", line 525, in irfft
    output = _raw_fft(a, n, axis, True, False, norm, out=out)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py", line 60, in _raw_fft
    raise ValueError(f"Invalid number of FFT data points ({n}) specified.")
ValueError: Invalid number of FFT data points (0) specified.
Falsifying example: test_irfft_produces_real_output(
    x_list=[0.0],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py:60
```
</details>

## Reproducing the Bug

```python
import numpy as np

print("=== Single-Element Array Bug Reproduction ===")
print()

# Bug 1: irfft on single-element array
print("Test 1: irfft with single-element array")
print("-" * 40)
x = np.array([1.0])
print(f"Input array x: {x}")
rfft_result = np.fft.rfft(x)
print(f"rfft(x) result: {rfft_result}")
try:
    result = np.fft.irfft(rfft_result)
    print(f"irfft(rfft_result): {result}")
except ValueError as e:
    print(f"ERROR: {e}")
print()

# Bug 2: irfft2 on single-element 2D array
print("Test 2: irfft2 with single-element 2D array")
print("-" * 40)
x2d = np.array([[1.0]])
print(f"Input array x2d: {x2d}")
rfft2_result = np.fft.rfft2(x2d)
print(f"rfft2(x2d) result: {rfft2_result}")
try:
    result2 = np.fft.irfft2(rfft2_result)
    print(f"irfft2(rfft2_result): {result2}")
except ValueError as e:
    print(f"ERROR: {e}")
print()

# Bug 3: irfftn on single-element array
print("Test 3: irfftn with single-element array")
print("-" * 40)
x = np.array([1.0])
print(f"Input array x: {x}")
rfftn_result = np.fft.rfftn(x)
print(f"rfftn(x) result: {rfftn_result}")
try:
    result3 = np.fft.irfftn(rfftn_result)
    print(f"irfftn(rfftn_result): {result3}")
except ValueError as e:
    print(f"ERROR: {e}")
print()

# Bug 4: hfft on single-element array
print("Test 4: hfft with single-element hermitian array")
print("-" * 40)
x_hermitian = np.array([1.0+0j])
print(f"Input array x_hermitian: {x_hermitian}")
try:
    result4 = np.fft.hfft(x_hermitian)
    print(f"hfft(x_hermitian): {result4}")
except ValueError as e:
    print(f"ERROR: {e}")
print()

print("=== Testing Workarounds ===")
print()

# Test workarounds
print("Workaround 1: irfft with n=1")
print("-" * 40)
x = np.array([1.0])
rfft_result = np.fft.rfft(x)
result = np.fft.irfft(rfft_result, n=1)
print(f"irfft(rfft([1.0]), n=1) = {result}")
print()

print("Workaround 2: irfft2 with s=(1,1)")
print("-" * 40)
x2d = np.array([[1.0]])
rfft2_result = np.fft.rfft2(x2d)
result2 = np.fft.irfft2(rfft2_result, s=(1, 1))
print(f"irfft2(rfft2([[1.0]]), s=(1,1)) = {result2}")
print()

print("Workaround 3: irfftn with s=(1,)")
print("-" * 40)
x = np.array([1.0])
rfftn_result = np.fft.rfftn(x)
result3 = np.fft.irfftn(rfftn_result, s=(1,))
print(f"irfftn(rfftn([1.0]), s=(1,)) = {result3}")
print()

print("Workaround 4: hfft with n=1")
print("-" * 40)
x_hermitian = np.array([1.0+0j])
result4 = np.fft.hfft(x_hermitian, n=1)
print(f"hfft([1.0+0j], n=1) = {result4}")
print()

print("=== Testing Round-Trip Property ===")
print()

print("Round-trip test: Does irfft(rfft(x)) == x?")
print("-" * 40)
# Test with 2-element array (should work)
x2 = np.array([1.0, 2.0])
print(f"2-element array: {x2}")
round_trip2 = np.fft.irfft(np.fft.rfft(x2))
print(f"irfft(rfft(x)): {round_trip2}")
print(f"Round-trip works? {np.allclose(x2, round_trip2)}")
print()

# Test with 1-element array (fails without n parameter)
x1 = np.array([1.0])
print(f"1-element array: {x1}")
try:
    round_trip1 = np.fft.irfft(np.fft.rfft(x1))
    print(f"irfft(rfft(x)): {round_trip1}")
    print(f"Round-trip works? {np.allclose(x1, round_trip1)}")
except ValueError as e:
    print(f"Round-trip FAILS with error: {e}")
    # Try with explicit n
    round_trip1_fixed = np.fft.irfft(np.fft.rfft(x1), n=1)
    print(f"With n=1: irfft(rfft(x), n=1) = {round_trip1_fixed}")
    print(f"Round-trip works with n=1? {np.allclose(x1, round_trip1_fixed)}")
```

<details>

<summary>
ValueError: Invalid number of FFT data points (0) specified
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/30/repo.py:84: DeprecationWarning: `axes` should not be `None` if `s` is not `None` (Deprecated in NumPy 2.0). In a future version of NumPy, this will raise an error and `s[i]` will correspond to the size along the transformed axis specified by `axes[i]`. To retain current behaviour, pass a sequence [0, ..., k-1] to `axes` for an array of dimension k.
  result3 = np.fft.irfftn(rfftn_result, s=(1,))
=== Single-Element Array Bug Reproduction ===

Test 1: irfft with single-element array
----------------------------------------
Input array x: [1.]
rfft(x) result: [1.+0.j]
ERROR: Invalid number of FFT data points (0) specified.

Test 2: irfft2 with single-element 2D array
----------------------------------------
Input array x2d: [[1.]]
rfft2(x2d) result: [[1.+0.j]]
ERROR: Invalid number of FFT data points (0) specified.

Test 3: irfftn with single-element array
----------------------------------------
Input array x: [1.]
rfftn(x) result: [1.+0.j]
ERROR: Invalid number of FFT data points (0) specified.

Test 4: hfft with single-element hermitian array
----------------------------------------
Input array x_hermitian: [1.+0.j]
ERROR: Invalid number of FFT data points (0) specified.

=== Testing Workarounds ===

Workaround 1: irfft with n=1
----------------------------------------
irfft(rfft([1.0]), n=1) = [1.]

Workaround 2: irfft2 with s=(1,1)
----------------------------------------
irfft2(rfft2([[1.0]]), s=(1,1)) = [[1.]]

Workaround 3: irfftn with s=(1,)
----------------------------------------
irfftn(rfftn([1.0]), s=(1,)) = [1.]

Workaround 4: hfft with n=1
----------------------------------------
hfft([1.0+0j], n=1) = [1.]

=== Testing Round-Trip Property ===

Round-trip test: Does irfft(rfft(x)) == x?
----------------------------------------
2-element array: [1. 2.]
irfft(rfft(x)): [1. 2.]
Round-trip works? True

1-element array: [1.]
Round-trip FAILS with error: Invalid number of FFT data points (0) specified.
With n=1: irfft(rfft(x), n=1) = [1.]
Round-trip works with n=1? True
```
</details>

## Why This Is A Bug

This violates the fundamental round-trip property of FFT operations. The documentation for `irfftn` explicitly states that `irfftn(rfftn(a), a.shape) == a` should hold within numerical accuracy. Similarly, for `irfft`, the documentation states this inverse relationship should work. However, for single-element arrays, the inverse functions fail completely without specifying the size parameter.

The root cause is the default size calculation formula used by these functions:
- `irfft`: `n = 2 * (a.shape[axis] - 1)`
- `irfftn`: `s[-1] = (a.shape[axes[-1]] - 1) * 2`
- `hfft`: `n = (a.shape[axis] - 1) * 2`

For single-element inputs (length 1), this formula produces `n = 2 * (1 - 1) = 0`, which is an invalid FFT size. The FFT implementation correctly rejects this with "Invalid number of FFT data points (0) specified."

Single-element FFTs are mathematically valid - the FFT of a single element is just that element itself. The forward transforms (`rfft`, `rfft2`, `rfftn`) handle single-element arrays correctly, but their inverse counterparts fail, breaking the expected inverse relationship.

## Relevant Context

The bug affects all inverse real FFT functions in numpy that use the default size calculation:
- `/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py:524` - `irfft` function
- `/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py:626` - `hfft` function
- `/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py:727` - `_cook_nd_args` function (used by `irfftn` and `irfft2`)

The documentation for these functions can be found at:
- https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html
- https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft2.html
- https://numpy.org/doc/stable/reference/generated/numpy.fft.irfftn.html
- https://numpy.org/doc/stable/reference/generated/numpy.fft.hfft.html

While single-element FFTs are an edge case, numpy is a fundamental scientific computing library where mathematical correctness is essential. The fix is simple and safe, maintaining backward compatibility while fixing the edge case.

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
@@ -724,7 +724,7 @@ def _cook_nd_args(a, s=None, axes=None, invreal=0):
     if len(s) != len(axes):
         raise ValueError("Shape and axes have different lengths.")
     if invreal and shapeless:
-        s[-1] = (a.shape[axes[-1]] - 1) * 2
+        s[-1] = max(1, (a.shape[axes[-1]] - 1) * 2)
     if None in s:
         msg = ("Passing an array containing `None` values to `s` is "
                "deprecated in NumPy 2.0 and will raise an error in "
```