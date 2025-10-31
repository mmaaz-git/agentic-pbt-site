# Bug Report: scipy.signal.windows.flattop Violates Normalization Contract

**Target**: `scipy.signal.windows.flattop`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `flattop` window function produces values that exceed the documented maximum of 1.0 by approximately 3 parts per billion, violating its API contract that promises "the maximum value normalized to 1".

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.signal import windows


window_functions_no_params = [
    'boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
    'blackmanharris', 'flattop', 'bartlett', 'barthann',
    'hamming', 'cosine', 'hann', 'lanczos', 'tukey'
]


@given(
    window_name=st.sampled_from(window_functions_no_params),
    M=st.integers(min_value=1, max_value=10000)
)
@settings(max_examples=500)
def test_normalization_property(window_name, M):
    window = windows.get_window(window_name, M, fftbins=True)
    max_val = np.max(np.abs(window))
    assert max_val <= 1.0 + 1e-10, f"{window_name} with M={M} has max value {max_val} > 1.0"

if __name__ == "__main__":
    test_normalization_property()
```

<details>

<summary>
**Failing input**: `window_name='flattop', M=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 24, in <module>
    test_normalization_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 14, in test_normalization_property
    window_name=st.sampled_from(window_functions_no_params),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 21, in test_normalization_property
    assert max_val <= 1.0 + 1e-10, f"{window_name} with M={M} has max value {max_val} > 1.0"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: flattop with M=2 has max value 1.000000003 > 1.0
Falsifying example: test_normalization_property(
    window_name='flattop',
    M=2,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.signal import windows

# Test case that demonstrates the bug
window = windows.flattop(3, sym=True)
max_val = np.max(window)

print(f"flattop(3, sym=True) = {window}")
print(f"max value = {max_val:.15f}")
print(f"exceeds 1.0? {max_val > 1.0}")

# Additional test cases
print("\nAdditional test cases:")
for M in [2, 3, 4, 5, 10]:
    for sym in [True, False]:
        window = windows.flattop(M, sym=sym)
        max_val = np.max(window)
        if max_val > 1.0:
            print(f"  M={M}, sym={sym}: max = {max_val:.15f} (exceeds 1.0)")

# Verify coefficients sum
coefficients = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
coeff_sum = sum(coefficients)
print(f"\nCoefficients sum = {coeff_sum:.15f}")
print(f"Exceeds 1.0 by: {coeff_sum - 1.0:.15e}")

# This should fail according to the documentation
assert max_val <= 1.0, f"Maximum value {max_val} exceeds documented limit of 1.0"
```

<details>

<summary>
AssertionError: Maximum value 1.000000003 exceeds documented limit of 1.0
</summary>
```
flattop(3, sym=True) = [-4.21051e-04  1.00000e+00 -4.21051e-04]
max value = 1.000000003000000
exceeds 1.0? True

Additional test cases:
  M=2, sym=False: max = 1.000000003000000 (exceeds 1.0)
  M=3, sym=True: max = 1.000000003000000 (exceeds 1.0)
  M=4, sym=False: max = 1.000000003000000 (exceeds 1.0)
  M=5, sym=True: max = 1.000000003000000 (exceeds 1.0)
  M=10, sym=False: max = 1.000000003000000 (exceeds 1.0)

Coefficients sum = 1.000000003000000
Exceeds 1.0 by: 3.000000026176508e-09
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/repo.py", line 28, in <module>
    assert max_val <= 1.0, f"Maximum value {max_val} exceeds documented limit of 1.0"
           ^^^^^^^^^^^^^^
AssertionError: Maximum value 1.000000003 exceeds documented limit of 1.0
```
</details>

## Why This Is A Bug

The documentation for `scipy.signal.windows.flattop` explicitly states in the Returns section:

> w : ndarray
>     The window, with the maximum value normalized to 1

This is an unambiguous API contract that guarantees the maximum value of the returned window will not exceed 1.0. However, the implementation violates this contract by producing values of 1.000000003 for various window sizes.

The root cause is that the flattop window is implemented as a weighted sum of cosine terms using these coefficients:
```python
a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
```

These coefficients sum to 1.000000003, exceeding 1.0 by approximately 3e-09. When the window is computed at positions where all cosine terms equal 1 (e.g., at n=0 when sym=False, or at the center position for odd M when sym=True), the window value becomes exactly the sum of all coefficients, which exceeds the documented limit.

While the excess is extremely small (3 parts per billion), it technically violates the documented normalization contract. This could cause issues in signal processing applications that:
- Rely on strict value bounds for numerical stability
- Use assertions or validation that expect values ≤ 1.0
- Perform calculations where accumulated floating-point errors could be amplified

## Relevant Context

The flattop window is described in the documentation as being "used for taking accurate measurements of signal amplitude in the frequency domain, with minimal scalloping error". This suggests that normalization accuracy is important for its intended use case.

The implementation is based on coefficients from "Digital Signal Processing for Measurement Systems" by D'Antona and Ferrero (2006), which describes this as a 5th-order cosine window optimized for minimal scalloping error. The slight coefficient sum excess appears to be due to rounding in the published values.

Source code location: `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:669-675`

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.flattop.html

## Proposed Fix

Normalize the coefficients to sum to exactly 1.0 while preserving their relative weights:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -668,7 +668,8 @@ def flattop(M, sym=True, *, xp=None, device=None):
     """
     xp = _namespace(xp)
     a = xp.asarray(
-        [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368],
+        # Normalized coefficients to sum to exactly 1.0
+        [0.21557894935326313, 0.41663157875010526, 0.27726315716821054, 0.08357894674926315, 0.006947367979157896],
         dtype=xp.float64, device=device
     )
     device = xp_device(a)
```