# Bug Report: scipy.signal.windows.tukey Symmetry Violation

**Target**: `scipy.signal.windows.tukey`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `tukey` window function violates its symmetry guarantee when `sym=True` for very small `alpha` values due to catastrophic cancellation in floating-point arithmetic.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as w

@given(
    M=st.integers(min_value=2, max_value=500),
    alpha=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=300)
def test_tukey_symmetry(M, alpha):
    window = w.tukey(M, alpha, sym=True)
    assert np.allclose(window, window[::-1]), \
        f"tukey({M}, {alpha}, sym=True) is not symmetric"
```

**Failing input**: `M=2, alpha=1.3824172351637151e-303`

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as w

M = 2
alpha = 1e-300

window = w.tukey(M, alpha, sym=True)
print(f"tukey({M}, {alpha}, sym=True) = {window}")
print(f"Reversed: {window[::-1]}")
print(f"Symmetric: {np.allclose(window, window[::-1])}")
```

Output:
```
tukey(2, 1e-300, sym=True) = [0. 1.]
Reversed: [1. 0.]
Symmetric: False
```

## Why This Is A Bug

The documentation states that when `sym=True`, the function "generates a symmetric window, for use in filter design." This is a fundamental property that should hold for all valid parameter values. However, for very small alpha values (e.g., 1e-300), the window is asymmetric.

The root cause is catastrophic cancellation in the floating-point computation of `w3`:

```python
w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
```

When `alpha` is very small (e.g., 1e-300), the terms `-2.0/alpha` and `2.0*n3/alpha/(M-1)` become extremely large (~1e300). Due to Python's left-to-right evaluation of addition, the expression is computed as:
1. `-2.0/alpha + 1` ≈ `-2e300` (the +1 is lost to floating-point precision)
2. `(-2e300) + 2e300` = 0

This yields an argument of 0 to cosine instead of π, resulting in `w3[0] = 1.0` instead of the correct value `0.0`, breaking symmetry.

## Fix

The fix is to reorder the arithmetic operations to avoid catastrophic cancellation by grouping the large terms first:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -943,7 +943,7 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):

     w1 = 0.5 * (1 + xp.cos(xp.pi * (-1 + 2.0*n1/alpha/(M-1))))
     w2 = xp.ones(n2.shape, device=device)
-    w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
+    w3 = 0.5 * (1 + xp.cos(xp.pi * (1 + (-2.0/alpha + 2.0*n3/alpha/(M-1)))))

     w = xp.concat((w1, w2, w3))