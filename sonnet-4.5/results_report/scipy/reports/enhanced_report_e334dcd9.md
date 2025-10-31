# Bug Report: scipy.signal.deconvolve Crashes on Leading Zeros in Divisor

**Target**: `scipy.signal.deconvolve`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `scipy.signal.deconvolve` function crashes with a ValueError when the divisor array contains leading zero coefficients, preventing deconvolution of valid polynomial/signal representations where higher-order terms are zero.

## Property-Based Test

```python
import numpy as np
import scipy.signal
from hypothesis import given, strategies as st, settings, assume

@given(
    divisor=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
    quotient=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=20)
)
@settings(max_examples=500)
def test_deconvolve_convolve_inverse(divisor, quotient):
    divisor_arr = np.array(divisor)
    quotient_arr = np.array(quotient)

    assume(np.any(np.abs(divisor_arr) > 1e-10))

    signal = scipy.signal.convolve(divisor_arr, quotient_arr, mode='full')
    recovered_quotient, remainder = scipy.signal.deconvolve(signal, divisor_arr)

    np.testing.assert_allclose(recovered_quotient, quotient_arr, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(remainder, 0, atol=1e-8)

# Run the test
if __name__ == "__main__":
    test_deconvolve_convolve_inverse()
```

<details>

<summary>
**Failing input**: `divisor=[0.0, 1.0], quotient=[0.0]`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 24, in <module>
  |     test_deconvolve_convolve_inverse()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 6, in test_deconvolve_convolve_inverse
  |     divisor=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 19, in test_deconvolve_convolve_inverse
    |     np.testing.assert_allclose(recovered_quotient, quotient_arr, rtol=1e-5, atol=1e-8)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1708, in assert_allclose
    |     assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
    |     ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                          verbose=verbose, header=header, equal_nan=equal_nan,
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                          strict=strict)
    |                          ^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 814, in assert_array_compare
    |     flagged |= func_assert_same_pos(x, y,
    |                ~~~~~~~~~~~~~~~~~~~~^^^^^^
    |                                     func=lambda xy: xy == -inf,
    |                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                                     hasval='-inf')
    |                                     ^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 777, in func_assert_same_pos
    |     raise AssertionError(msg)
    | AssertionError:
    | Not equal to tolerance rtol=1e-05, atol=1e-08
    |
    | -inf location mismatch:
    |  ACTUAL: array([ 1.000000e+000,  0.000000e+000,  1.467430e+077, -2.153351e+154,
    |         3.159892e+231,           -inf])
    |  DESIRED: array([1., 1., 0., 0., 0., 0.])
    | Falsifying example: test_deconvolve_convolve_inverse(
    |     divisor=[2.0443903514663864e-77, 3.0],
    |     quotient=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1085
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:771
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 19, in test_deconvolve_convolve_inverse
    |     np.testing.assert_allclose(recovered_quotient, quotient_arr, rtol=1e-5, atol=1e-8)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1708, in assert_allclose
    |     assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
    |     ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                          verbose=verbose, header=header, equal_nan=equal_nan,
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                          strict=strict)
    |                          ^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 916, in assert_array_compare
    |     raise AssertionError(msg)
    | AssertionError:
    | Not equal to tolerance rtol=1e-05, atol=1e-08
    |
    | Mismatched elements: 1 / 2 (50%)
    | Max absolute difference among violations: 1.
    | Max relative difference among violations: 1.
    |  ACTUAL: array([1., 0.])
    |  DESIRED: array([1., 1.])
    | Falsifying example: test_deconvolve_convolve_inverse(
    |     divisor=[2.0443903514663864e-77, 1.0],
    |     quotient=[1.0, 1.0],
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:600
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:964
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1016
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1021
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:3048
    |         (and 5 more with settings.verbosity >= verbose)
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 17, in test_deconvolve_convolve_inverse
    |     recovered_quotient, remainder = scipy.signal.deconvolve(signal, divisor_arr)
    |                                     ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/signal/_signaltools.py", line 2471, in deconvolve
    |     quot = lfilter(num, den, input)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/signal/_signaltools.py", line 2303, in lfilter
    |     result =_sigtools._linear_filter(b, a, x, axis)
    | ValueError: BUG: filter coefficient a[0] == 0 not supported yet
    | Falsifying example: test_deconvolve_convolve_inverse(
    |     divisor=[0.0, 1.0],
    |     quotient=[0.0],  # or any other generated value
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.signal

divisor = np.array([0.0, 1.0])
signal = np.array([0.0, 5.0])

quotient, remainder = scipy.signal.deconvolve(signal, divisor)
print(f"Quotient: {quotient}")
print(f"Remainder: {remainder}")
```

<details>

<summary>
ValueError: BUG: filter coefficient a[0] == 0 not supported yet
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/repo.py", line 7, in <module>
    quotient, remainder = scipy.signal.deconvolve(signal, divisor)
                          ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/signal/_signaltools.py", line 2471, in deconvolve
    quot = lfilter(num, den, input)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/signal/_signaltools.py", line 2303, in lfilter
    result =_sigtools._linear_filter(b, a, x, axis)
ValueError: BUG: filter coefficient a[0] == 0 not supported yet
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **The error message explicitly acknowledges this is a bug**: The ValueError message literally states "BUG: filter coefficient a[0] == 0 not supported yet", indicating the developers recognize this as a deficiency that should be fixed, not intended behavior.

2. **The documented mathematical contract is violated**: According to the function's docstring, `deconvolve` should return quotient and remainder such that `signal = convolve(divisor, quotient) + remainder`. This mathematical property should hold for any divisor with non-zero coefficients somewhere in the array, regardless of leading zeros.

3. **Leading zeros represent valid polynomial/filter structures**: In polynomial and signal processing contexts, an array like `[0.0, 1.0]` represents a valid polynomial (0*x^1 + 1*x^0 = 1) or a valid filter response. The leading zero simply indicates that the higher-order term has a coefficient of zero, which is mathematically legitimate.

4. **The workaround proves the operation is valid**: When we manually strip the leading zero and call `deconvolve(signal, [1.0])`, the function works correctly and produces the expected result. This demonstrates that the underlying mathematical operation is both valid and implementable.

5. **Inconsistency with documented behavior**: The docstring states that `deconvolve` "performs polynomial division (same operation, but also accepts poly1d objects)" similar to `numpy.polydiv`. While `numpy.polydiv` also struggles with this case (producing NaN), at least it attempts the operation rather than crashing with an error.

## Relevant Context

The bug occurs in the internal call chain when `deconvolve` calls `lfilter` with the divisor as the denominator coefficient. The `lfilter` function (and its underlying C implementation `_sigtools._linear_filter`) requires that the first coefficient be non-zero for normalization purposes.

The issue can be found in `/home/npc/.local/lib/python3.13/site-packages/scipy/signal/_signaltools.py:2471` where `lfilter` is called without preprocessing the divisor to handle leading zeros.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.deconvolve.html

## Proposed Fix

The fix involves stripping leading zeros from the divisor before calling `lfilter`. This maintains the mathematical correctness while avoiding the internal limitation:

```diff
--- a/scipy/signal/_signaltools.py
+++ b/scipy/signal/_signaltools.py
@@ -2463,6 +2463,16 @@ def deconvolve(signal, divisor):
         raise ValueError("divisor must be 1-D.")
     N = num.shape[0]
     D = den.shape[0]
+
+    # Strip leading zeros from divisor to avoid lfilter limitation
+    # Find first non-zero coefficient
+    nonzero_indices = xp.nonzero(xp.abs(den) > 0)[0]
+    if len(nonzero_indices) == 0:
+        raise ValueError("divisor must have at least one non-zero coefficient")
+    first_nonzero = nonzero_indices[0]
+    if first_nonzero > 0:
+        den = den[first_nonzero:]
+        D = den.shape[0]
     if D > N:
         quot = []
         rem = num
```