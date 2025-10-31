# Bug Report: scipy.signal.windows.tukey produces NaN with extremely small alpha

**Target**: `scipy.signal.windows.tukey`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.signal.windows.tukey(M, alpha)` produces NaN values when alpha is extremely small (< ~1e-307). While this is a rare edge case, the function should either handle it gracefully or document the minimum alpha requirement.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import scipy.signal.windows as windows


@given(st.integers(min_value=1, max_value=100),
       st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_tukey_no_nan(M, alpha):
    """Tukey window should never produce NaN for alpha in [0, 1]."""
    w = windows.tukey(M, alpha)
    assert not np.any(np.isnan(w)), \
        f"tukey({M}, alpha={alpha}) produced NaN: {w}"
```

**Failing input**: `M=2, alpha=2.225e-311`

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as windows

test_cases = [
    (2, 1e-308),
    (5, 1e-308),
    (10, 1e-308),
]

for M, alpha in test_cases:
    w = windows.tukey(M, alpha)
    print(f"tukey({M}, alpha={alpha:.2e}) = {w}")

# Output:
# tukey(2, alpha=1.00e-308) = [ 0. nan]
# tukey(5, alpha=1.00e-308) = [ 0.  1.  1.  1. nan]
# tukey(10, alpha=1.00e-308) = [ 0.  1.  1.  1.  1.  1.  1.  1.  1. nan]
```

## Why This Is A Bug

The Tukey window implementation divides by alpha in its formula:

```python
w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
```

When alpha < ~1e-307:
1. `2.0/alpha` overflows to infinity
2. `cos(infinity)` evaluates to NaN
3. This NaN propagates to the output

The issue affects the last element(s) of the window. While alpha < 1e-307 is extremely rare in practice, the function should either:
- Add a minimum alpha validation check
- Use a more numerically stable formula for small alpha
- Document the minimum alpha requirement

## Fix

Add validation to reject extremely small alpha values:

```diff
diff --git a/scipy/signal/windows/_windows.py b/scipy/signal/windows/_windows.py
index 1234567..89abcdef 100644
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -918,6 +918,10 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):
     if alpha <= 0:
         return xp.ones(M, dtype=xp.float64, device=device)
     elif alpha >= 1.0:
         return hann(M, sym=sym, xp=xp, device=device)
+    elif alpha < 1e-300:
+        # For extremely small alpha, avoid numerical overflow in division
+        # alpha→0 should approach boxcar (all ones)
+        return xp.ones(M, dtype=xp.float64, device=device)

     M, needs_trunc = _extend(M, sym)
```