# Bug Report: scipy.signal.windows.gaussian produces NaN with std=0

**Target**: `scipy.signal.windows.gaussian`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.signal.windows.gaussian(M, std=0.0)` produces NaN values for all odd M >= 3, causing silent data corruption. The function should either reject std=0 with a clear error message or handle it gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import scipy.signal.windows as windows


@given(st.integers(min_value=1, max_value=100),
       st.floats(min_value=-100, max_value=100, allow_nan=False))
def test_gaussian_no_nan(M, std):
    """Gaussian window should never produce NaN for finite std values."""
    if std <= 0:
        # Either raise an error or produce valid output
        try:
            w = windows.gaussian(M, std)
            assert not np.any(np.isnan(w)), \
                f"gaussian({M}, std={std}) produced NaN"
        except (ValueError, ZeroDivisionError):
            pass
    else:
        w = windows.gaussian(M, std)
        assert not np.any(np.isnan(w))
```

**Failing input**: `M=3, std=0.0` (and all odd M >= 3)

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as windows

for M in [3, 5, 7, 9, 11]:
    w = windows.gaussian(M, std=0.0)
    print(f"gaussian({M}, std=0.0) = {w}")

# Output:
# gaussian(3, std=0.0) = [ 0. nan  0.]
# gaussian(5, std=0.0) = [ 0.  0. nan  0.  0.]
# gaussian(7, std=0.0) = [ 0.  0.  0. nan  0.  0.  0.]
# gaussian(9, std=0.0) = [ 0.  0.  0.  0. nan  0.  0.  0.  0.]
# gaussian(11, std=0.0) = [ 0.  0.  0.  0.  0. nan  0.  0.  0.  0.  0.]
```

## Why This Is A Bug

The function accepts std=0 without validation but produces NaN values due to a 0/0 division in the Gaussian formula. Specifically:

1. For odd M with sym=True, the center index has n=0
2. With std=0, sig2 = 2*std² = 0
3. The formula computes exp(-n²/sig2) = exp(0/0) = exp(NaN) = NaN

This violates basic expectations:
- Input validation should catch invalid parameters before computation
- Mathematical functions should not silently produce NaN for seemingly reasonable inputs
- std=0 represents a degenerate case (zero-width Gaussian) that should be explicitly handled

## Fix

Add input validation to reject std <= 0:

```diff
diff --git a/scipy/signal/windows/_windows.py b/scipy/signal/windows/_windows.py
index 1234567..89abcdef 100644
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -1445,6 +1445,9 @@ def gaussian(M, std, sym=True, *, xp=None, device=None):
     """
     xp = _namespace(xp)

+    if std <= 0:
+        raise ValueError("std must be positive, got std={}".format(std))
+
     if _len_guards(M):
         return xp.ones(M, dtype=xp.float64, device=device)
     M, needs_trunc = _extend(M, sym)
```