# Bug Report: scipy.signal.windows.tukey NaN Values for Very Small Alpha

**Target**: `scipy.signal.windows.tukey`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `tukey` window function produces NaN values when the `alpha` parameter is very small (but greater than 0). This occurs due to numerical overflow in the computation `-2.0/alpha` when alpha approaches the smallest positive float.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from scipy.signal import windows

@given(M=st.integers(min_value=2, max_value=500),
       alpha=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_tukey_nonnegative(M, alpha):
    w = windows.tukey(M, alpha)
    assert np.all(w >= 0), f"Tukey window should be non-negative"
```

**Failing input**: `M=2, alpha=5e-324`

## Reproducing the Bug

```python
import numpy as np
from scipy.signal import windows

M = 2
alpha = 5e-324

w = windows.tukey(M, alpha)
print(f"tukey({M}, {alpha}) = {w}")
print(f"Contains NaN: {np.any(np.isnan(w))}")
```

Output:
```
tukey(2, 5e-324) = [ 0. nan]
Contains NaN: True
```

## Why This Is A Bug

The `tukey` function's docstring states that `alpha` represents "the fraction of the window inside the cosine tapered region" and that "If zero, the Tukey window is equivalent to a rectangular window."

While the code has a guard for `alpha <= 0` (line 931), it does not handle the case where `alpha` is very small but still positive. When `alpha` is extremely small (e.g., `5e-324`), the expression `-2.0/alpha` in line 946 causes numerical overflow:

```python
w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
```

This overflow produces infinity, and `cos(infinity)` yields NaN, resulting in invalid window values.

A user would reasonably expect that any alpha in the valid range [0, 1] should produce a valid window, with very small alpha values behaving similarly to alpha=0 (a rectangular window).

## Fix

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -928,7 +928,10 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):
     if _len_guards(M):
         return xp.ones(M, dtype=xp.float64, device=device)

-    if alpha <= 0:
+    # Treat very small alpha as 0 to avoid numerical overflow in division
+    # The threshold 1e-10 is chosen to be well above machine epsilon but
+    # small enough that such alpha values are effectively 0 in practice
+    if alpha <= 1e-10:
         return xp.ones(M, dtype=xp.float64, device=device)
     elif alpha >= 1.0:
         return hann(M, sym=sym, xp=xp, device=device)
```