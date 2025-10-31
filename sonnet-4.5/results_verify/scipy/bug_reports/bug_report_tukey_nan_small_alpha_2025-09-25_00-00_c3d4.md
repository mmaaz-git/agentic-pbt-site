# Bug Report: tukey Window Produces NaN for Very Small Positive Alpha Values

**Target**: `scipy.signal.windows.tukey`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `tukey` window function produces NaN values when given extremely small positive alpha values due to numerical overflow when dividing by alpha. This violates the expectation that all window values should be finite.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.signal.windows as windows

@given(st.integers(min_value=1, max_value=1000), st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=200)
def test_tukey_alpha_produces_finite_values(M, alpha):
    w = windows.tukey(M, alpha=alpha)
    assert np.all(np.isfinite(w))
```

**Failing input**: `M=2, alpha=2.225073858507e-311`

## Reproducing the Bug

```python
import scipy.signal.windows as windows
import numpy as np

w = windows.tukey(2, alpha=2.225073858507e-311)
print(f"Result: {w}")
print(f"Has NaN: {np.any(np.isnan(w))}")
print(f"Expected: all finite values")
```

Output:
```
Result: [ 0. nan]
Has NaN: True
Expected: all finite values
```

## Why This Is A Bug

1. **Produces invalid output**: Window functions should always produce finite values for valid inputs. Alpha in the range [0.0, 1.0] is documented as valid.

2. **Numerical instability**: The implementation divides by alpha (lines 79 and 81 in the source), which causes overflow when alpha is extremely small but positive:
   ```python
   w1 = 0.5 * (1 + xp.cos(xp.pi * (-1 + 2.0*n1/alpha/(M-1))))
   w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
   ```

3. **Inconsistent handling**: The function checks `alpha <= 0` on line 66 but doesn't handle the case where alpha is positive but so small that division by it causes overflow.

## Fix

The function should treat very small positive alpha values the same as alpha=0:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -911,7 +911,8 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):
     if _len_guards(M):
         return xp.ones(M, dtype=xp.float64, device=device)

-    if alpha <= 0:
+    # Treat very small alpha as 0 to avoid numerical overflow
+    if alpha <= 0 or alpha < 1e-10:
         return xp.ones(M, dtype=xp.float64, device=device)
     elif alpha >= 1.0:
         return hann(M, sym=sym, xp=xp, device=device)
```

Alternatively, use a more robust check:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -1,4 +1,5 @@
 import math
+import sys

 @@ -911,7 +912,8 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):
     if _len_guards(M):
         return xp.ones(M, dtype=xp.float64, device=device)

-    if alpha <= 0:
+    # Treat alpha too small for safe division as 0
+    if alpha <= 0 or alpha <= sys.float_info.epsilon * 100:
         return xp.ones(M, dtype=xp.float64, device=device)
     elif alpha >= 1.0:
         return hann(M, sym=sym, xp=xp, device=device)
```