# Bug Report: scipy.fft.irfft Fails on Single-Element Arrays with Misleading Error Message

**Target**: `scipy.fft.irfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.fft.irfft` raises a misleading ValueError when processing the output of `scipy.fft.rfft` for single-element arrays without an explicit `n` parameter, breaking the fundamental FFT round-trip property that `irfft(rfft(x)) ≈ x`.

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
def test_rfft_irfft_round_trip(data):
    x = np.array(data)
    result = scipy.fft.irfft(scipy.fft.rfft(x))
    assert np.allclose(result, x, rtol=1e-10, atol=1e-12)

if __name__ == "__main__":
    test_rfft_irfft_round_trip()
```

<details>

<summary>
**Failing input**: `data=[0.0]` (single-element array)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 24, in <module>
  |     test_rfft_irfft_round_trip()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 7, in test_rfft_irfft_round_trip
  |     st.lists(
  |
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 21, in test_rfft_irfft_round_trip
    |     assert np.allclose(result, x, rtol=1e-10, atol=1e-12)
    |            ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_rfft_irfft_round_trip(
    |     data=[0.0, 0.0, 0.0, 0.0, 0.0, 9743.0],
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 21, in test_rfft_irfft_round_trip
    |     assert np.allclose(result, x, rtol=1e-10, atol=1e-12)
    |            ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py", line 2365, in allclose
    |     res = all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
    |               ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py", line 2496, in isclose
    |     result = (less_equal(abs(x - y), atol + rtol * abs(y))
    |                              ~~^~~
    | ValueError: operands could not be broadcast together with shapes (2,) (3,)
    | Falsifying example: test_rfft_irfft_round_trip(
    |     data=[0.0, 0.0, 0.0],
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 20, in test_rfft_irfft_round_trip
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
    | Falsifying example: test_rfft_irfft_round_trip(
    |     data=[0.0],
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/.local/lib/python3.13/site-packages/scipy/fft/_pocketfft/basic.py:90
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.fft

# Test case with single-element array
x = np.array([5.0])
print(f"Input array: {x}")
print(f"Input shape: {x.shape}")

# Perform rfft
rfft_out = scipy.fft.rfft(x)
print(f"\nrfft output: {rfft_out}")
print(f"rfft output shape: {rfft_out.shape}")

# Try to perform irfft without specifying n (this should fail)
print("\nAttempting irfft without n parameter...")
try:
    irfft_out = scipy.fft.irfft(rfft_out)
    print(f"irfft output: {irfft_out}")
except ValueError as e:
    print(f"ERROR: {e}")

# Now try with n specified
print("\nAttempting irfft with n=1...")
irfft_out_with_n = scipy.fft.irfft(rfft_out, n=1)
print(f"irfft output with n=1: {irfft_out_with_n}")
print(f"Matches original? {np.allclose(irfft_out_with_n, x)}")
```

<details>

<summary>
ValueError: Invalid number of data points (0) specified
</summary>
```
Input array: [5.]
Input shape: (1,)

rfft output: [5.+0.j]
rfft output shape: (1,)

Attempting irfft without n parameter...
ERROR: Invalid number of data points (0) specified

Attempting irfft with n=1...
irfft output with n=1: [5.]
Matches original? True
```
</details>

## Why This Is A Bug

This issue violates the fundamental FFT round-trip property and presents three distinct problems:

1. **Misleading error message**: The error states "Invalid number of data points (0) specified" when the user never specified n=0. The library internally computed this value using the formula `n = 2*(m-1)` where m is the rfft output length. For single-element arrays, m=1, resulting in n=0.

2. **Broken round-trip property**: The property-based test reveals that `irfft(rfft(x))` fails not only for single-element arrays but also for all odd-length arrays when n is not explicitly specified. This breaks the expected mathematical invariant that the inverse transform should recover the original signal.

3. **Inconsistent behavior**: Even-length arrays work correctly without specifying n, while odd-length arrays (including single elements) require explicit n parameter. This inconsistency is not clearly documented and creates confusion.

## Relevant Context

The bug occurs in `/home/npc/.local/lib/python3.13/site-packages/scipy/fft/_pocketfft/basic.py:87-90` where the default value of n is computed:

```python
if n is None:
    n = (tmp.shape[axis] - 1) * 2
    if n < 1:
        raise ValueError(f"Invalid number of data points ({n}) specified")
```

The formula `n = (tmp.shape[axis] - 1) * 2` assumes even-length output, which is why:
- Even-length arrays (e.g., [1,2]) work: rfft produces 2 elements, n = (2-1)*2 = 2 ✓
- Odd-length arrays (e.g., [1,2,3]) fail: rfft produces 2 elements, n = (2-1)*2 = 2 ≠ 3
- Single elements fail catastrophically: rfft produces 1 element, n = (1-1)*2 = 0

The scipy documentation states that for odd-length signals, the n parameter must be specified, but the error message doesn't help users understand this requirement.

## Proposed Fix

The fix should improve the error message to guide users when the computed n is invalid:

```diff
--- a/scipy/fft/_pocketfft/basic.py
+++ b/scipy/fft/_pocketfft/basic.py
@@ -86,8 +86,12 @@ def c2r(forward, x, n=None, axis=-1, norm=None, overwrite_x=False,
     # Last axis utilizes hermitian symmetry
     if n is None:
         n = (tmp.shape[axis] - 1) * 2
         if n < 1:
-            raise ValueError(f"Invalid number of data points ({n}) specified")
+            raise ValueError(
+                f"Cannot infer output length from rfft/hfft input of length "
+                f"{tmp.shape[axis]}. For odd-length or single-element arrays, "
+                f"specify the output length explicitly with the 'n' parameter."
+            )
     else:
         tmp, _ = _fix_shape_1d(tmp, (n//2) + 1, axis)
```