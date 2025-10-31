# Bug Report: scipy.signal.windows.tukey NaN for Small Alpha Values

**Target**: `scipy.signal.windows.tukey`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `tukey` window function returns NaN values for very small but valid alpha parameters (approximately < 1e-300) due to numerical overflow when dividing by alpha.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.signal.windows as windows

@given(st.integers(min_value=1, max_value=100), st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=300)
def test_tukey_alpha_range(M, alpha):
    result = windows.tukey(M, alpha=alpha)
    assert len(result) == M
    assert np.all(np.isfinite(result))
```

**Failing input**: `M=2, alpha=2.225073858507e-311`

## Reproducing the Bug

```python
import numpy as np
from scipy.signal.windows import tukey

result = tukey(2, alpha=2.225073858507e-311)
print(f"Result: {result}")
print(f"Contains NaN: {np.any(np.isnan(result))}")
```

Output:
```
Result: [ 0. nan]
Contains NaN: True
```

## Why This Is A Bug

The `tukey` function's docstring states that alpha is a "float" parameter representing "the fraction of the window inside the cosine tapered region," with special cases:
- "If zero, the Tukey window is equivalent to a rectangular window."
- "If one, the Tukey window is equivalent to a Hann window."

The docstring implies alpha can be any float in [0, 1], but does not specify a minimum positive value. However, for extremely small alpha values (< ~1e-300), the function returns NaN due to overflow in the expression `-2.0/alpha`, which becomes `-inf`, and `cos(-inf)` produces NaN.

## Fix

The issue occurs in the calculation for the third region of the window at line 946 of `_windows.py`:

```python
w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
```

When alpha is very small, `-2.0/alpha` overflows. The fix should either:

1. Add input validation to reject alpha values below a reasonable threshold (e.g., `1e-300`), or
2. Add special handling for very small alpha values to avoid the overflow

Suggested fix:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -920,6 +920,11 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):
     if _len_guards(M):
         return xp.ones(M, dtype=xp.float64, device=device)

+    # Guard against very small alpha values that cause overflow
+    if alpha < 1e-300:
+        # For extremely small alpha, behave as boxcar (alpha=0)
+        alpha = 0.0
+
     if alpha <= 0:
         return xp.ones(M, dtype=xp.float64, device=device)
     elif alpha >= 1.0:
```

Alternatively, validate the input:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -920,6 +920,9 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):
     if _len_guards(M):
         return xp.ones(M, dtype=xp.float64, device=device)

+    if 0 < alpha < 1e-300:
+        raise ValueError(f"alpha must be 0 or >= 1e-300 to avoid numerical overflow, got {alpha}")
+
     if alpha <= 0:
         return xp.ones(M, dtype=xp.float64, device=device)
     elif alpha >= 1.0:
```