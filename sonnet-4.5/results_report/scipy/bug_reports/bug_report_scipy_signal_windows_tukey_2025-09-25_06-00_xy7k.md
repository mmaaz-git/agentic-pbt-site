# Bug Report: scipy.signal.windows.tukey NaN Values with Tiny Alpha

**Target**: `scipy.signal.windows.tukey`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `tukey` window function produces NaN values when called with very small but valid alpha parameter values (e.g., alpha < 1e-300), instead of returning a valid window array.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from scipy.signal.windows import tukey


@given(st.integers(min_value=2, max_value=100),
       st.floats(min_value=1e-320, max_value=1e-100, allow_nan=False, allow_infinity=False))
def test_tukey_no_nan_with_tiny_alpha(M, alpha):
    w = tukey(M, alpha=alpha, sym=True)

    assert len(w) == M
    assert np.all(np.isfinite(w)), \
        f"tukey({M}, alpha={alpha}) contains non-finite values: {w}"
```

**Failing input**: `M=2, alpha=1e-320`

## Reproducing the Bug

```python
import numpy as np
from scipy.signal.windows import tukey

w = tukey(2, alpha=1e-309, sym=True)
print(f"tukey(2, alpha=1e-309) = {w}")
print(f"Contains NaN: {np.any(np.isnan(w))}")

w = tukey(10, alpha=1e-309, sym=True)
print(f"tukey(10, alpha=1e-309) = {w}")
print(f"Contains NaN: {np.any(np.isnan(w))}")
```

Output:
```
tukey(2, alpha=1e-309) = [ 0. nan]
Contains NaN: True
tukey(10, alpha=1e-309) = [ 0.  1.  1.  1.  1.  1.  1.  1.  1. nan]
Contains NaN: True
```

## Why This Is A Bug

The `tukey` function's docstring specifies that `alpha` is a float parameter representing "the fraction of the window inside the cosine tapered region" with valid range [0, 1]. Very small positive values like 1e-309 are within this range and should be handled gracefully. Instead, the function produces NaN values due to numerical overflow in the internal computation.

The root cause is in line 946 of `_windows.py`:
```python
w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
```

When `alpha` is extremely small, `-2.0/alpha` overflows to infinity, causing `cos(infinity)` to return NaN.

This violates the function's contract of returning a valid window array and can silently corrupt downstream calculations when NaN values propagate through the user's code.

## Fix

Add a guard to treat extremely small alpha values as alpha=0 (rectangular window), similar to the existing guard for alpha <= 0. This is numerically sound since alpha values below a certain threshold (e.g., 1e-100) are indistinguishable from zero in the window's behavior.

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -928,7 +928,7 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):
     if _len_guards(M):
         return xp.ones(M, dtype=xp.float64, device=device)

-    if alpha <= 0:
+    if alpha <= 0 or alpha < 1e-100:
         return xp.ones(M, dtype=xp.float64, device=device)
     elif alpha >= 1.0:
         return hann(M, sym=sym, xp=xp, device=device)
```

Alternatively, a more robust fix would be to clamp the cosine argument to avoid overflow:
```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -941,9 +941,13 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):
     n2 = n[width+1:M-width-1]
     n3 = n[M-width-1:]

-    w1 = 0.5 * (1 + xp.cos(xp.pi * (-1 + 2.0*n1/alpha/(M-1))))
+    arg1 = xp.pi * (-1 + 2.0*n1/alpha/(M-1))
+    w1 = 0.5 * (1 + xp.cos(arg1))
     w2 = xp.ones(n2.shape, device=device)
-    w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
+    arg3 = xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))
+    arg3 = xp.clip(arg3, -1e15, 1e15)
+    w3 = 0.5 * (1 + xp.cos(arg3))

     w = xp.concat((w1, w2, w3))
```

The first fix is simpler and more aligned with the function's existing structure. The second fix provides more gradual degradation but may not be compatible with all array namespaces.