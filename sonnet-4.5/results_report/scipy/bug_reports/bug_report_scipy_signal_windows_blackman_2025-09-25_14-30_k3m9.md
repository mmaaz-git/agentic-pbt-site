# Bug Report: scipy.signal.windows.blackman Negative Endpoint Values

**Target**: `scipy.signal.windows.blackman`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `blackman` window function returns negative values at its endpoints due to floating point rounding errors, violating the expected property that window values should be non-negative.

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
```

**Failing input**: `M=2` (or any M ≥ 2)

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as w

window = w.blackman(10)
print(f"min value = {window.min():.20e}")
print(f"has negative values = {np.any(window < 0)}")
print(f"endpoints = [{window[0]:.20e}, {window[-1]:.20e}]")

a0, a1, a2 = 0.42, 0.5, 0.08
theoretical_endpoint = a0 - a1 + a2
print(f"\nTheoretical endpoint value: {theoretical_endpoint}")
```

Output:
```
min value = -1.38777878078144567553e-17
has negative values = True
endpoints = [-1.38777878078144567553e-17, -1.38777878078144567553e-17]

Theoretical endpoint value: 0.0
```

## Why This Is A Bug

The Blackman window is defined as: `w(n) = 0.42 - 0.5*cos(2πn/(M-1)) + 0.08*cos(4πn/(M-1))`

At the endpoints (n=0 and n=M-1):
- `w = 0.42 - 0.5*cos(0) + 0.08*cos(0) = 0.42 - 0.5 + 0.08 = 0.0`

The theoretical value is exactly 0.0, but due to floating point rounding in the cosine calculations, the actual value becomes a tiny negative number (-1.38e-17).

This bug affects:
- **All M values ≥ 2**: Always produces exactly 2 negative values at indices 0 and M-1
- **Code that assumes non-negativity**: Any code using `if window[i] > 0` or relying on non-negative values will behave incorrectly
- **Consistency**: Other similar windows (blackmanharris, nuttall) don't have this issue

## Fix

```diff
diff --git a/scipy/signal/windows/_windows.py b/scipy/signal/windows/_windows.py
index abc123..def456 100644
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -blackman function
     w = (a[0] - a[1] * xp.cos(fac) + a[2] * xp.cos(2 * fac))
+    # Clip to avoid negative values from floating point errors
+    w = xp.clip(w, 0, None)

     return w
```

The fix is to clip the window values to be non-negative, which corrects the floating point error while preserving the intended zero endpoints.