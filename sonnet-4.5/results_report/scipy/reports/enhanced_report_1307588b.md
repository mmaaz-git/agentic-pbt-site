# Bug Report: scipy.signal.deconvolve Violates Mathematical Invariant Due to Numerical Instability

**Target**: `scipy.signal.deconvolve`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.signal.deconvolve` violates its documented mathematical property that `signal = convolve(divisor, quotient) + remainder` for certain divisor patterns, producing reconstruction errors exceeding 6% due to severe numerical instability.

## Property-Based Test

```python
from hypothesis import given, settings, assume, strategies as st
import numpy as np
import scipy.signal

@settings(max_examples=200)
@given(
    signal=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=10, max_size=50),
    divisor=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=3, max_size=10)
)
def test_deconvolve_inverse_property(signal, divisor):
    signal_arr = np.array(signal, dtype=np.float64)
    divisor_arr = np.array(divisor, dtype=np.float64)

    assume(np.abs(divisor_arr[0]) > 1e-10)

    quotient, remainder = scipy.signal.deconvolve(signal_arr, divisor_arr)

    reconstructed = scipy.signal.convolve(divisor_arr, quotient, mode='full')
    reconstructed_with_remainder = reconstructed.copy()
    min_len = min(len(reconstructed_with_remainder), len(remainder))
    reconstructed_with_remainder[:min_len] += remainder[:min_len]

    expected_len = len(signal_arr)
    reconstructed_trimmed = reconstructed_with_remainder[:expected_len]

    np.testing.assert_allclose(signal_arr, reconstructed_trimmed, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    test_deconvolve_inverse_property()
```

<details>

<summary>
**Failing input**: `signal=[0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 1.0, 0.0], divisor=[1e-05, 1.0, 0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 29, in <module>
    test_deconvolve_inverse_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 6, in test_deconvolve_inverse_property
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 26, in test_deconvolve_inverse_property
    np.testing.assert_allclose(signal_arr, reconstructed_trimmed, rtol=1e-5, atol=1e-8)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1708, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         verbose=verbose, header=header, equal_nan=equal_nan,
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         strict=strict)
                         ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 916, in assert_array_compare
    raise AssertionError(msg)
AssertionError:
Not equal to tolerance rtol=1e-05, atol=1e-08

Mismatched elements: 1 / 10 (10%)
Max absolute difference among violations: 1.
Max relative difference among violations: inf
 ACTUAL: array([ 0.,  0.,  0.,  0.,  0., 10.,  0.,  0.,  1.,  0.])
 DESIRED: array([ 0.,  0.,  0.,  0.,  0., 10.,  0.,  0.,  0.,  0.])
Falsifying example: test_deconvolve_inverse_property(
    signal=[0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 1.0, 0.0],
    divisor=[1e-05, 1.0, 0.0],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:964
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1016
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1021
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:102
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:862
        (and 3 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.signal

signal = np.array([65.0] + [0.0]*21 + [1.875])
divisor = np.array([0.125, 0.0, 2.0])

quotient, remainder = scipy.signal.deconvolve(signal, divisor)
reconstructed = scipy.signal.convolve(divisor, quotient, mode='full')
reconstructed[:len(remainder)] += remainder

print(f"Signal length: {len(signal)}")
print(f"Signal: {signal}")
print(f"Divisor: {divisor}")
print(f"Quotient length: {len(quotient)}")
print(f"Remainder length: {len(remainder)}")
print(f"Reconstructed length: {len(reconstructed)}")
print()
print(f"Expected last value: {signal[-1]}")
print(f"Reconstructed last value: {reconstructed[len(signal)-1]}")
print(f"Error: {abs(signal[-1] - reconstructed[len(signal)-1])}")
print(f"Relative error: {abs(signal[-1] - reconstructed[len(signal)-1])/abs(signal[-1])*100:.2f}%")
print(f"Max quotient magnitude: {np.max(np.abs(quotient)):.2e}")
print()
print("Full reconstruction comparison:")
print(f"Original signal: {signal}")
print(f"Reconstructed (trimmed): {reconstructed[:len(signal)]}")
print(f"Difference: {signal - reconstructed[:len(signal)]}")
print(f"Max absolute error: {np.max(np.abs(signal - reconstructed[:len(signal)]))}")
```

<details>

<summary>
Numerical instability causes 6.67% reconstruction error
</summary>
```
Signal length: 23
Signal: [65.     0.     0.     0.     0.     0.     0.     0.     0.     0.
  0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
  0.     0.     1.875]
Divisor: [0.125 0.    2.   ]
Quotient length: 21
Remainder length: 23
Reconstructed length: 23

Expected last value: 1.875
Reconstructed last value: 2.0
Error: 0.125
Relative error: 6.67%
Max quotient magnitude: 5.72e+14

Full reconstruction comparison:
Original signal: [65.     0.     0.     0.     0.     0.     0.     0.     0.     0.
  0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
  0.     0.     1.875]
Reconstructed (trimmed): [65.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  2.]
Difference: [ 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
  0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
  0.     0.    -0.125]
Max absolute error: 0.125
```
</details>

## Why This Is A Bug

The `scipy.signal.deconvolve` function's documentation explicitly guarantees that "signal = convolve(divisor, quotient) + remainder". This is a fundamental mathematical property that users rely upon for signal processing applications. However, this property is violated with significant errors (up to 6.67% relative error in the test case) due to numerical instability.

The function uses `lfilter` internally to compute the deconvolution, which can become numerically unstable when the divisor represents an unstable filter. The divisor `[0.125, 0.0, 2.0]` corresponds to the transfer function H(z) = 1/(0.125 + 2z²), which has poles at z = ±0.25j√2, leading to exponential growth of intermediate values (quotient magnitudes reach 5.72×10¹⁴) and subsequent catastrophic cancellation errors.

The bug is particularly problematic because:
1. The function silently returns incorrect results without any warning
2. The inputs are valid, well-formed floating-point numbers
3. The divisor's leading coefficient is non-zero as required
4. The failure occurs with moderate-length signals (23 elements)
5. Users have no indication that the mathematical guarantee has been violated

## Relevant Context

The deconvolution implementation in scipy.signal uses linear filtering (`lfilter`) which treats the problem as an inverse filtering operation. The current implementation can be found in scipy/signal/_signaltools.py. The function does not check for numerical stability or validate that the reconstructed signal matches the input within reasonable tolerances.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.deconvolve.html

The issue affects any divisor that represents a marginally stable or unstable filter, particularly those with zeros near or outside the unit circle when viewed as a transfer function denominator.

## Proposed Fix

Add numerical stability checking and a verification step to warn users when the mathematical property is violated:

```diff
--- a/scipy/signal/_signaltools.py
+++ b/scipy/signal/_signaltools.py
@@ -1234,6 +1234,7 @@ def deconvolve(signal, divisor):
     array([ 0.,  1.,  0.,  0.,  1.,  1.,  0.,  0.])

     """
+    import warnings
     xp = array_namespace(signal, divisor)

     num = xpx.atleast_nd(xp.asarray(signal), ndim=1, xp=xp)
@@ -1251,4 +1252,18 @@ def deconvolve(signal, divisor):
         input[0] = 1
         quot = lfilter(num, den, input)
         rem = num - convolve(den, quot, mode='full')
+
+        # Verify reconstruction property
+        reconstructed = convolve(den, quot, mode='full')
+        reconstructed[:len(rem)] += rem
+        reconstruction_error = xp.max(xp.abs(num - reconstructed[:len(num)]))
+        signal_scale = xp.max(xp.abs(num))
+
+        if signal_scale > 0 and reconstruction_error > 1e-8 * signal_scale:
+            warnings.warn(
+                f"Deconvolution may be numerically unstable. "
+                f"Reconstruction error: {float(reconstruction_error):.2e}, "
+                f"relative to signal scale: {float(reconstruction_error/signal_scale):.2e}",
+                RuntimeWarning
+            )
     return quot, rem
```