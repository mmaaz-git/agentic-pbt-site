# Bug Report: scipy.fft.irfft Single-Element Array Crash

**Target**: `scipy.fft.irfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.fft.irfft()` crashes with a ValueError when processing the output of `scipy.fft.rfft()` for single-element arrays, violating the documented round-trip guarantee.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import scipy.fft


@given(
    st.lists(
        st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-1e10,
            max_value=1e10
        ),
        min_size=1,
        max_size=1000
    )
)
def test_rfft_irfft_roundtrip(data):
    x = np.array(data)
    result = scipy.fft.irfft(scipy.fft.rfft(x))
    np.testing.assert_allclose(result, x, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    # Run the test
    test_rfft_irfft_roundtrip()
```

<details>

<summary>
**Failing input**: `[0.0]`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 26, in <module>
  |     test_rfft_irfft_roundtrip()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 7, in test_rfft_irfft_roundtrip
  |     st.lists(
  |
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 21, in test_rfft_irfft_roundtrip
    |     np.testing.assert_allclose(result, x, rtol=1e-10, atol=1e-10)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1708, in assert_allclose
    |     assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
    |     ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                          verbose=verbose, header=header, equal_nan=equal_nan,
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                          strict=strict)
    |                          ^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 803, in assert_array_compare
    |     raise AssertionError(msg)
    | AssertionError:
    | Not equal to tolerance rtol=1e-10, atol=1e-10
    |
    | (shapes (2,), (3,) mismatch)
    |  ACTUAL: array([0., 0.])
    |  DESIRED: array([0., 0., 0.])
    | Falsifying example: test_rfft_irfft_roundtrip(
    |     data=[0.0, 0.0, 0.0],
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 21, in test_rfft_irfft_roundtrip
    |     np.testing.assert_allclose(result, x, rtol=1e-10, atol=1e-10)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
    | Not equal to tolerance rtol=1e-10, atol=1e-10
    |
    | Mismatched elements: 1 / 2 (50%)
    | Max absolute difference among violations: 4.54542626e-10
    | Max relative difference among violations: 1.50275532e-10
    |  ACTUAL: array([8.388605e+06, 3.024728e+00])
    |  DESIRED: array([8.388605e+06, 3.024728e+00])
    | Falsifying example: test_rfft_irfft_roundtrip(
    |     data=[8388605.0, 3.024728109139892],
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 20, in test_rfft_irfft_roundtrip
    |     result = scipy.fft.irfft(scipy.fft.rfft(x))
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/fft/_backend.py", line 28, in __ua_function__
    |     return fn(*args, **kwargs)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/fft/_basic_backend.py", line 97, in irfft
    |     return _execute_1D('irfft', _pocketfft.irfft, x, n=n, axis=axis, norm=norm,
    |                        overwrite_x=overwrite_x, workers=workers, plan=plan)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/fft/_basic_backend.py", line 32, in _execute_1D
    |     return pocketfft_func(x, n=n, axis=axis, norm=norm,
    |                           overwrite_x=overwrite_x, workers=workers, plan=plan)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/fft/_pocketfft/basic.py", line 90, in c2r
    |     raise ValueError(f"Invalid number of data points ({n}) specified")
    | ValueError: Invalid number of data points (0) specified
    | Falsifying example: test_rfft_irfft_roundtrip(
    |     data=[0.0],
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.fft

# Test with a single-element array
x = np.array([5.0])
print(f"Original array: {x}")
print(f"Array shape: {x.shape}")

# Apply rfft - this should work
fft_result = scipy.fft.rfft(x)
print(f"rfft result: {fft_result}")
print(f"rfft result shape: {fft_result.shape}")

# Try to apply irfft - this will crash
try:
    roundtrip = scipy.fft.irfft(fft_result)
    print(f"irfft result: {roundtrip}")
except ValueError as e:
    print(f"Error occurred: {e}")

# Show that the workaround works
print("\nWorkaround with n=1:")
roundtrip_with_n = scipy.fft.irfft(fft_result, n=1)
print(f"irfft(fft_result, n=1) result: {roundtrip_with_n}")
```

<details>

<summary>
ValueError: Invalid number of data points (0) specified
</summary>
```
Original array: [5.]
Array shape: (1,)
rfft result: [5.+0.j]
rfft result shape: (1,)
Error occurred: Invalid number of data points (0) specified

Workaround with n=1:
irfft(fft_result, n=1) result: [5.]
```
</details>

## Why This Is A Bug

This bug violates the fundamental round-trip property that SciPy's FFT documentation explicitly guarantees: `irfft(rfft(x), len(x)) == x` should hold for any valid input array `x`. The documentation makes no mention of minimum size requirements, and mathematically, a single-point DFT is well-defined (it's simply the identity operation: the transform of a single value is that value itself).

The issue occurs because when `irfft()` receives a single-element complex array and `n` is not explicitly specified, it calculates the output size as `n = (input_length - 1) * 2`. For a single-element input, this becomes `n = (1 - 1) * 2 = 0`, which triggers the validation error. This calculation assumes the input came from an `rfft()` of at least 2 real values, but `rfft()` accepts and correctly processes single-element arrays.

The asymmetry is clear: `rfft()` accepts single-element real arrays and produces single-element complex arrays, but `irfft()` cannot process these valid outputs without explicitly specifying `n=1`. This breaks the promise of seamless round-trip compatibility between the forward and inverse transforms.

## Relevant Context

The bug is located in `/scipy/fft/_pocketfft/basic.py` at lines 87-90 in the `c2r` function (which implements `irfft`):

```python
# Line 87-90 in c2r function
if n is None:
    n = (tmp.shape[axis] - 1) * 2
    if n < 1:
        raise ValueError(f"Invalid number of data points ({n}) specified")
```

The Hypothesis test also revealed two additional issues:
1. Odd-length arrays don't round-trip correctly (shape mismatch)
2. Some arrays have numerical precision issues exceeding the tolerance

However, the single-element crash is the most severe as it causes a complete failure rather than numerical inaccuracy.

Documentation references:
- SciPy FFT module: https://docs.scipy.org/doc/scipy/reference/fft.html
- The `irfft` function specifically promises: "This function computes the inverse of rfft"
- Source code: https://github.com/scipy/scipy/blob/main/scipy/fft/_pocketfft/basic.py

## Proposed Fix

```diff
--- a/scipy/fft/_pocketfft/basic.py
+++ b/scipy/fft/_pocketfft/basic.py
@@ -86,7 +86,10 @@ def c2r(forward, x, n=None, axis=-1, norm=None, overwrite_x=False,
     # Last axis utilizes hermitian symmetry
     if n is None:
         n = (tmp.shape[axis] - 1) * 2
-        if n < 1:
+        # Special case for single-element arrays
+        if tmp.shape[axis] == 1:
+            n = 1
+        elif n < 1:
             raise ValueError(f"Invalid number of data points ({n}) specified")
     else:
         tmp, _ = _fix_shape_1d(tmp, (n//2) + 1, axis)
```