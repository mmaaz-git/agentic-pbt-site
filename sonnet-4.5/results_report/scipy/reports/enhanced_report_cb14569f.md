# Bug Report: scipy.signal.deconvolve Crashes with Zero Leading Coefficient

**Target**: `scipy.signal.deconvolve`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.signal.deconvolve` crashes with a ValueError when the divisor's leading coefficient is zero, even though this represents a mathematically valid polynomial division operation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy as np
import scipy.signal


@given(
    signal=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=50),
    divisor=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=30),
)
@settings(max_examples=500)
def test_deconvolve_round_trip(signal, divisor):
    assume(len(divisor) <= len(signal))
    assume(any(abs(d) > 1e-6 for d in divisor))

    signal_arr = np.array(signal)
    divisor_arr = np.array(divisor)

    quotient, remainder = scipy.signal.deconvolve(signal_arr, divisor_arr)

    reconstructed = scipy.signal.convolve(divisor_arr, quotient) + remainder

    np.testing.assert_allclose(reconstructed, signal_arr, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    # Run the test
    test_deconvolve_round_trip()
```

<details>

<summary>
**Failing input**: `signal=[0.0, 0.0], divisor=[0.0, 1.0]`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/1/hypo.py:20: RuntimeWarning: invalid value encountered in add
  reconstructed = scipy.signal.convolve(divisor_arr, quotient) + remainder
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 27, in <module>
  |     test_deconvolve_round_trip()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 7, in test_deconvolve_round_trip
  |     signal=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=50),
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 22, in test_deconvolve_round_trip
    |     np.testing.assert_allclose(reconstructed, signal_arr, rtol=1e-10, atol=1e-10)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1708, in assert_allclose
    |     assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
    |     ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                          verbose=verbose, header=header, equal_nan=equal_nan,
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                          strict=strict)
    |                          ^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 808, in assert_array_compare
    |     flagged = func_assert_same_pos(x, y, func=isnan, hasval='nan')
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 777, in func_assert_same_pos
    |     raise AssertionError(msg)
    | AssertionError:
    | Not equal to tolerance rtol=1e-10, atol=1e-10
    |
    | nan location mismatch:
    |  ACTUAL: array([ 1., nan, nan])
    |  DESIRED: array([1., 0., 0.])
    | Falsifying example: test_deconvolve_round_trip(
    |     signal=[1.0, 0.0, 0.0],
    |     divisor=[1.3845498899524102e-264, 1.0],
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:771
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 18, in test_deconvolve_round_trip
    |     quotient, remainder = scipy.signal.deconvolve(signal_arr, divisor_arr)
    |                           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/signal/_signaltools.py", line 2471, in deconvolve
    |     quot = lfilter(num, den, input)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/signal/_signaltools.py", line 2303, in lfilter
    |     result =_sigtools._linear_filter(b, a, x, axis)
    | ValueError: BUG: filter coefficient a[0] == 0 not supported yet
    | Falsifying example: test_deconvolve_round_trip(
    |     signal=[0.0, 0.0],  # or any other generated value
    |     divisor=[0.0, 1.0],
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.signal

signal = np.array([1.0, 2.0])
divisor = np.array([0.0, 1.0])

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
  File "/home/npc/pbt/agentic-pbt/worker_/1/repo.py", line 7, in <module>
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

This violates expected behavior for several critical reasons:

1. **The error message itself admits it's a bug**: The exception explicitly states "BUG: filter coefficient a[0] == 0 not supported yet". The use of the word "BUG" in production code indicates the developers acknowledge this as a known defect that should be fixed.

2. **Mathematically valid operation**: A divisor like `[0.0, 1.0]` represents the polynomial `0*x + 1 = 1`, which is simply the constant 1. Dividing any polynomial by a constant is a well-defined mathematical operation. The leading zero merely indicates the polynomial has lower degree than the array length suggests.

3. **Documentation provides no warning**: The function's docstring states it "returns the quotient and remainder such that `signal = convolve(divisor, quotient) + remainder`" without any precondition that `divisor[0]` must be non-zero. The documentation references `numpy.polydiv` as performing "the same operation", yet provides no warning about this limitation.

4. **Inconsistent with polynomial division semantics**: The function claims to perform polynomial division but fails on valid polynomial inputs that simply have leading zeros in their coefficient representation.

## Relevant Context

The bug occurs in the internal call chain at `/home/npc/.local/lib/python3.13/site-packages/scipy/signal/_signaltools.py:2471` where `deconvolve` calls `lfilter(num, den, input)`. The `lfilter` function requires `a[0] != 0` for numerical stability in its recursive filter implementation, but this is an implementation detail that shouldn't constrain the higher-level `deconvolve` API.

The Hypothesis test also revealed a second failure mode where very small (but non-zero) leading coefficients like `1.3845498899524102e-264` cause numerical instability, producing NaN values instead of correct results. This suggests the function has broader numerical issues with small leading coefficients.

For comparison, `numpy.polydiv` handles the same input without crashing (though it produces inf/nan warnings):
- Input: `[1.0, 2.0], [0.0, 1.0]`
- numpy.polydiv result: `Quotient = [inf], Remainder = [nan, -inf]`

While numpy.polydiv doesn't crash, a proper implementation should strip leading zeros to produce the correct result: `Quotient = [1.0, 2.0], Remainder = [0.0]`.

## Proposed Fix

```diff
--- a/scipy/signal/_signaltools.py
+++ b/scipy/signal/_signaltools.py
@@ -2476,6 +2476,12 @@ def deconvolve(signal, divisor):

     num = xpx.atleast_nd(xp.asarray(signal), ndim=1, xp=xp)
     den = xpx.atleast_nd(xp.asarray(divisor), ndim=1, xp=xp)
+
+    # Strip leading zeros from divisor to handle polynomial division correctly
+    nonzero_indices = xp.nonzero(den)[0]
+    if len(nonzero_indices) > 0:
+        den = den[nonzero_indices[0]:]
+
     if num.ndim > 1:
         raise ValueError("signal must be 1-D.")
     if den.ndim > 1:
```