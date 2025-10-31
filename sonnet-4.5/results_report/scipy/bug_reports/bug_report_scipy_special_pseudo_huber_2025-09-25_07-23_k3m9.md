# Bug Report: scipy.special.pseudo_huber NaN for Small Delta Values

**Target**: `scipy.special.pseudo_huber`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `pseudo_huber` function returns NaN instead of a finite value when `delta` is very small (approximately < 10^-190) due to numerical overflow in the intermediate computation.

## Property-Based Test

```python
import scipy.special as sp
import numpy as np
from hypothesis import given, strategies as st, settings

@settings(max_examples=2000)
@given(st.floats(allow_nan=False, allow_infinity=False, min_value=1e-308, max_value=10),
       st.floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=10))
def test_pseudo_huber_returns_finite(delta, r):
    result = sp.pseudo_huber(delta, r)
    assert np.isfinite(result), f"pseudo_huber({delta}, {r}) returned {result}, expected finite value"
```

**Failing input**: `delta=2.3581411596114265e-203, r=1.0` (and any delta < ~1e-190)

## Reproducing the Bug

```python
import scipy.special as sp

delta = 1e-200
r = 1.0

result = sp.pseudo_huber(delta, r)
print(f"sp.pseudo_huber({delta}, {r}) = {result}")
```

**Output**: `sp.pseudo_huber(1e-200, 1.0) = nan`

**Expected**: A finite positive value (approximately |r| - delta = 1.0 - 1e-200 ≈ 1.0)

## Why This Is A Bug

The pseudo-Huber loss function is defined as:

```
pseudo_huber(delta, r) = delta^2 * (sqrt(1 + (r/delta)^2) - 1)
```

This function should return a finite value for all positive `delta` and all finite `r`. However, when `delta` is very small and `r` is non-zero, the computation suffers from numerical overflow:

1. `(r/delta)^2` overflows to infinity when `delta` is very small
2. `sqrt(infinity) = infinity`
3. `infinity - 1 = infinity`
4. `delta^2 * infinity = 0 * infinity = NaN`

This violates the function's contract and mathematical properties. The pseudo-Huber loss should be well-defined and finite for all valid inputs.

## Fix

The implementation should use a numerically stable formula when `|r/delta|` is large. For large values of `|r/delta|`, the formula can be approximated as:

```
pseudo_huber(delta, r) ≈ |r| - delta
```

Alternatively, the formula can be rewritten to avoid overflow. When `|r/delta| >> 1`, we have:

```
delta^2 * (sqrt(1 + (r/delta)^2) - 1)
= delta^2 * |r/delta| * (1 + O(delta^2/r^2))
= delta * |r| * (1 + O(delta^2/r^2))
≈ |r| - delta/2 for large |r/delta|
```

A robust implementation would detect when `|r/delta|` is large enough to cause overflow and switch to the asymptotic formula:

```diff
def pseudo_huber(delta, r):
+   # For large |r/delta|, use asymptotic formula to avoid overflow
+   ratio_squared = (r / delta) ** 2
+   if np.isinf(ratio_squared):
+       return np.abs(r) - delta
+
    # Original formula
    return delta**2 * (np.sqrt(1 + (r/delta)**2) - 1)
```

Or more elegantly, restructure the formula to avoid computing `(r/delta)^2` explicitly when it would overflow.