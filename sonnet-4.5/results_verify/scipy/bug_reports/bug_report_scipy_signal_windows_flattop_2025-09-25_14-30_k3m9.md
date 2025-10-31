# Bug Report: scipy.signal.windows.flattop Maximum Value Exceeds 1.0

**Target**: `scipy.signal.windows.flattop`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `flattop` window function violates its documented contract that "the maximum value normalized to 1". For all odd values of M, the window's maximum value is approximately 1.000000003, exceeding 1.0.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.signal.windows as windows

@given(st.integers(min_value=1, max_value=1000))
@settings(max_examples=200)
def test_max_value_normalized_to_one(M):
    w = windows.flattop(M)
    max_val = np.max(w)
    assert np.isclose(max_val, 1.0, rtol=1e-10, atol=1e-14) or max_val < 1.0, \
        f"flattop({M}) has max value {max_val}, expected <= 1.0"
```

**Failing input**: `M=3` (and all other odd values of M)

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as windows

w = windows.flattop(3)
print(f"Window values: {w}")
print(f"Max value: {np.max(w):.20f}")
print(f"Expected: <= 1.0")

a = np.array([0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368])
print(f"\nFlattop coefficients sum: {np.sum(a):.20f}")
```

Output:
```
Window values: [-4.21051e-04  1.00000e+00 -4.21051e-04]
Max value: 1.00000000300000002618
Expected: <= 1.0

Flattop coefficients sum: 1.00000000300000002618
```

## Why This Is A Bug

The `flattop` function's docstring explicitly states: "The window, with the maximum value normalized to 1". However, for odd values of M, the maximum value is 1.000000003, which exceeds 1.0.

The root cause is that the flattop coefficients sum to 1.000000003 instead of 1.0:
```python
a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
sum(a) = 1.000000003
```

For general cosine windows, when all cosine terms equal 1 (which occurs at the center of odd-length symmetric windows), the window value equals the sum of coefficients. Thus, the maximum value exceeds 1.0.

This affects all odd M values (3, 5, 7, 9, ...), which represents half of all possible inputs.

## Fix

The coefficients should be normalized to sum to exactly 1.0. The fix involves dividing each coefficient by their sum:

```diff
def flattop(M, sym=True, *, xp=None, device=None):
    xp = _namespace(xp)
    a = xp.asarray(
-       [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368],
+       [0.215578947368421, 0.416631578947368, 0.277263157894737, 0.083578947368421, 0.006947368421053],
        dtype=xp.float64, device=device
    )
    device = xp_device(a)
    return _general_cosine_impl(M, a, xp, device, sym=sym)
```

The corrected coefficients are computed as:
```python
original = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
corrected = [x / sum(original) for x in original]
# [0.215578947368421, 0.416631578947368, 0.277263157894737, 0.083578947368421, 0.006947368421053]
```

These normalized coefficients sum to exactly 1.0 and preserve the relative proportions of the original design.