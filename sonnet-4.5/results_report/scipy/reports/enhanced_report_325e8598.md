# Bug Report: scipy.signal.windows.blackman Negative Endpoint Values Due to Floating Point Errors

**Target**: `scipy.signal.windows.blackman`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `blackman` window function returns small negative values at its endpoints (-1.39e-17) instead of the mathematically expected zero, due to floating point representation errors when computing 0.42 - 0.50 + 0.08.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as w

@given(st.integers(min_value=2, max_value=1000))
@settings(max_examples=500)
def test_blackman_non_negative(M):
    """Window values should be non-negative."""
    result = w.blackman(M)
    assert np.all(result >= 0), \
        f"blackman(M={M}) has negative values: min={np.min(result)}"

if __name__ == "__main__":
    # Run the test
    test_blackman_non_negative()
```

<details>

<summary>
**Failing input**: `M=2` (or any M >= 2)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 15, in <module>
    test_blackman_non_negative()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 6, in test_blackman_non_negative
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 10, in test_blackman_non_negative
    assert np.all(result >= 0), \
           ~~~~~~^^^^^^^^^^^^^
AssertionError: blackman(M=2) has negative values: min=-1.3877787807814457e-17
Falsifying example: test_blackman_non_negative(
    M=2,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as w

# Test with various M values
for M in [2, 5, 10, 100, 1000]:
    window = w.blackman(M)
    min_val = window.min()
    has_negative = np.any(window < 0)
    negative_count = np.sum(window < 0)

    print(f"M={M}:")
    print(f"  min value = {min_val:.20e}")
    print(f"  has negative values = {has_negative}")
    print(f"  number of negative values = {negative_count}")
    if has_negative:
        negative_indices = np.where(window < 0)[0]
        print(f"  negative value indices = {negative_indices.tolist()}")
        print(f"  negative values = {[f'{window[i]:.20e}' for i in negative_indices]}")
    print(f"  endpoints = [{window[0]:.20e}, {window[-1]:.20e}]")
    print()

# Verify the mathematical expectation
print("Mathematical analysis:")
a0, a1, a2 = 0.42, 0.50, 0.08
theoretical_endpoint = a0 - a1*1.0 + a2*1.0  # cos(0) = 1 for both terms
print(f"Theoretical endpoint value: {theoretical_endpoint}")
print(f"This should be exactly 0.0")
print()

# Check what the actual implementation does
print("What the implementation computes at endpoints:")
M = 10
n = 0  # First endpoint
term1 = 0.42
term2 = -0.50 * np.cos(0)  # cos(0) = 1
term3 = 0.08 * np.cos(0)   # cos(0) = 1
endpoint_value = term1 + term2 + term3
print(f"At n=0: 0.42 - 0.50*cos(0) + 0.08*cos(0) = {term1} + {term2} + {term3} = {endpoint_value}")
print(f"Due to floating point representation: {0.42 - 0.50 + 0.08}")
```

<details>

<summary>
Demonstrates negative values at endpoints for all M >= 2
</summary>
```
M=2:
  min value = -1.38777878078144567553e-17
  has negative values = True
  number of negative values = 2
  negative value indices = [0, 1]
  negative values = ['-1.38777878078144567553e-17', '-1.38777878078144567553e-17']
  endpoints = [-1.38777878078144567553e-17, -1.38777878078144567553e-17]

M=5:
  min value = -1.38777878078144567553e-17
  has negative values = True
  number of negative values = 2
  negative value indices = [0, 4]
  negative values = ['-1.38777878078144567553e-17', '-1.38777878078144567553e-17']
  endpoints = [-1.38777878078144567553e-17, -1.38777878078144567553e-17]

M=10:
  min value = -1.38777878078144567553e-17
  has negative values = True
  number of negative values = 2
  negative value indices = [0, 9]
  negative values = ['-1.38777878078144567553e-17', '-1.38777878078144567553e-17']
  endpoints = [-1.38777878078144567553e-17, -1.38777878078144567553e-17]

M=100:
  min value = -1.38777878078144567553e-17
  has negative values = True
  number of negative values = 2
  negative value indices = [0, 99]
  negative values = ['-1.38777878078144567553e-17', '-1.38777878078144567553e-17']
  endpoints = [-1.38777878078144567553e-17, -1.38777878078144567553e-17]

M=1000:
  min value = -1.38777878078144567553e-17
  has negative values = True
  number of negative values = 2
  negative value indices = [0, 999]
  negative values = ['-1.38777878078144567553e-17', '-1.38777878078144567553e-17']
  endpoints = [-1.38777878078144567553e-17, -1.38777878078144567553e-17]

Mathematical analysis:
Theoretical endpoint value: -1.3877787807814457e-17
This should be exactly 0.0

What the implementation computes at endpoints:
At n=0: 0.42 - 0.50*cos(0) + 0.08*cos(0) = 0.42 + -0.5 + 0.08 = -1.3877787807814457e-17
Due to floating point representation: -1.3877787807814457e-17
```
</details>

## Why This Is A Bug

The Blackman window is mathematically defined as:
```
w(n) = 0.42 - 0.5 * cos(2πn/(M-1)) + 0.08 * cos(4πn/(M-1))
```

At the endpoints (n=0 and n=M-1), the cosine terms evaluate to 1, giving:
```
w(0) = w(M-1) = 0.42 - 0.5*1 + 0.08*1 = 0.42 - 0.5 + 0.08 = 0.0
```

However, due to floating point representation, the sum `0.42 - 0.5 + 0.08` doesn't equal exactly 0.0 but instead produces -1.3877787807814457e-17. This violates the documented and expected property that window functions should produce non-negative values.

The issue occurs because:
1. The coefficients 0.42, 0.5, and 0.08 cannot be represented exactly in binary floating point
2. The arithmetic operations accumulate rounding errors
3. The result is a tiny negative value instead of the mathematically expected zero

This affects all M values >= 2, always producing exactly 2 negative values at indices 0 and M-1.

## Relevant Context

The blackman window function is implemented in `/scipy/signal/windows/_windows.py` at lines 399-486. It uses the `_general_cosine_impl` helper function with coefficients `[0.42, 0.50, 0.08]`.

The documentation at line 429 states the mathematical formula but doesn't mention that the implementation may produce tiny negative values at the endpoints.

Other similar window functions in the same module (blackmanharris, nuttall) use 4 terms which may balance out better, though they could potentially have similar issues.

SciPy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.blackman.html

## Proposed Fix

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -60,6 +60,9 @@ def _general_cosine_impl(M, a, xp, device, sym=True):
     w = xp.zeros(M, dtype=xp.float64, device=device)
     for k in range(a.shape[0]):
         w += a[k] * xp.cos(k * fac)
+
+    # Clip to avoid negative values from floating point errors at endpoints
+    w = xp.clip(w, 0, None)

     return _truncate(w, needs_trunc)
```

Alternatively, the fix could be applied specifically in the blackman function after line 486, though fixing it in `_general_cosine_impl` would handle similar issues in all cosine-based windows.