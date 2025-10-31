# Bug Report: numpy.fft.hfft Single-Element Array Crash

**Target**: `numpy.fft.hfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.fft.hfft` crashes with ValueError when given a single-element array due to a flaw in the default parameter calculation that produces n=0. The same issue affects `numpy.fft.irfft`.

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
    # If we get here without exception, the test passes
    assert result is not None

if __name__ == "__main__":
    test_hfft_single_element_crash()
```

<details>

<summary>
**Failing input**: `value=0.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 16, in <module>
    test_hfft_single_element_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 7, in test_hfft_single_element_crash
    min_value=-1e6, max_value=1e6))
    ^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 11, in test_hfft_single_element_crash
    result = numpy.fft.hfft(a)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py", line 628, in hfft
    output = irfft(conjugate(a), n, axis, norm=new_norm, out=None)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py", line 525, in irfft
    output = _raw_fft(a, n, axis, True, False, norm, out=out)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py", line 60, in _raw_fft
    raise ValueError(f"Invalid number of FFT data points ({n}) specified.")
ValueError: Invalid number of FFT data points (0) specified.
Falsifying example: test_hfft_single_element_crash(
    value=0.0,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np

# Demonstrate the crash with numpy.fft.hfft on single-element array
a = np.array([1.0])
try:
    result = np.fft.hfft(a)
    print(f"Success: hfft([1.0]) = {result}")
except ValueError as e:
    print(f"Error with hfft([1.0]): {e}")

# Show that the workaround works
try:
    result_with_n = np.fft.hfft(a, n=2)
    print(f"Success with explicit n: hfft([1.0], n=2) = {result_with_n}")
except ValueError as e:
    print(f"Error with hfft([1.0], n=2): {e}")

# Test other FFT functions for comparison
print("\nComparison with other FFT functions:")
functions_to_test = [
    ('fft', np.fft.fft),
    ('ifft', np.fft.ifft),
    ('rfft', np.fft.rfft),
    ('irfft', np.fft.irfft),
    ('hfft', np.fft.hfft),
    ('ihfft', np.fft.ihfft)
]

for name, func in functions_to_test:
    try:
        result = func(a)
        print(f"{name}: SUCCESS - returns {result}")
    except ValueError as e:
        print(f"{name}: FAILS - {e}")
```

<details>

<summary>
ValueError: Invalid number of FFT data points (0) specified
</summary>
```
Error with hfft([1.0]): Invalid number of FFT data points (0) specified.
Success with explicit n: hfft([1.0], n=2) = [1. 1.]

Comparison with other FFT functions:
fft: SUCCESS - returns [1.+0.j]
ifft: SUCCESS - returns [1.+0.j]
rfft: SUCCESS - returns [1.+0.j]
irfft: FAILS - Invalid number of FFT data points (0) specified.
hfft: FAILS - Invalid number of FFT data points (0) specified.
ihfft: SUCCESS - returns [1.-0.j]
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Inconsistent API Behavior**: Out of 6 FFT functions in numpy.fft, 4 handle single-element arrays correctly (fft, ifft, rfft, ihfft), while only 2 fail (hfft, irfft). This breaks the principle of least surprise for users expecting consistent behavior across the FFT API.

2. **Mathematical Edge Case Not Handled**: The documented formula `n = 2*(m-1)` in both hfft and irfft produces mathematically invalid input when m=1 (resulting in n=0). The code at line 626 of _pocketfft.py blindly applies this formula without checking for edge cases.

3. **Contradicts Documentation**: The numpy.fft.hfft documentation states that n defaults to `2*(m-1)` but nowhere mentions that single-element arrays are unsupported or that n=0 is invalid. The documentation for the `n` parameter only states it's the "Length of the transformed axis of the output."

4. **Underlying Algorithm Supports It**: The fact that `np.fft.hfft([1.0], n=2)` works successfully and returns `[1., 1.]` proves the underlying FFT algorithm can handle single-element inputs - it's purely the default parameter calculation that's broken.

5. **Unhelpful Error Message**: The error "Invalid number of FFT data points (0) specified" doesn't help users understand that the issue is with single-element arrays or how to fix it.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/fft/_pocketfft.py` at:
- Line 626 for hfft: `n = (a.shape[axis] - 1) * 2`
- Line 524 for irfft: `n = (a.shape[axis] - 1) * 2`

Both functions use the same flawed formula that produces n=0 for single-element inputs, which then gets passed to `_raw_fft` which raises the ValueError at line 60.

NumPy FFT documentation: https://numpy.org/doc/stable/reference/routines.fft.html

The error traceback shows the call path:
1. hfft() calculates n=0 at line 626
2. Calls irfft() at line 628 with n=0
3. irfft() calls _raw_fft() at line 525 with n=0
4. _raw_fft() raises ValueError at line 60

## Proposed Fix

```diff
--- a/numpy/fft/_pocketfft.py
+++ b/numpy/fft/_pocketfft.py
@@ -521,7 +521,11 @@ def irfft(a, n=None, axis=-1, norm=None, out=None):
     """
     a = asarray(a)
     if n is None:
-        n = (a.shape[axis] - 1) * 2
+        m = a.shape[axis]
+        if m == 1:
+            n = 1  # or n = 2 for even-length default
+        else:
+            n = (m - 1) * 2
     output = _raw_fft(a, n, axis, True, False, norm, out=out)
     return output

@@ -623,7 +627,11 @@ def hfft(a, n=None, axis=-1, norm=None, out=None):
     """
     a = asarray(a)
     if n is None:
-        n = (a.shape[axis] - 1) * 2
+        m = a.shape[axis]
+        if m == 1:
+            n = 1  # or n = 2 for even-length default
+        else:
+            n = (m - 1) * 2
     new_norm = _swap_direction(norm)
     output = irfft(conjugate(a), n, axis, norm=new_norm, out=None)
     return output
```