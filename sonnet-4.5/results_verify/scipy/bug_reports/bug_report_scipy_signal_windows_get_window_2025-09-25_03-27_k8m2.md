# Bug Report: scipy.signal.windows.get_window Default Parameter Inconsistency

**Target**: `scipy.signal.windows.get_window`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `get_window` function has inconsistent default parameters compared to the window functions it wraps. Specifically, `get_window('hann', M)` returns a different result than `hann(M)` because `get_window` defaults to `fftbins=True` (periodic window) while individual window functions default to `sym=True` (symmetric window).

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as windows


@given(st.integers(min_value=2, max_value=100))
@settings(max_examples=300)
def test_get_window_consistency(M):
    for window_name in ['hann', 'hamming', 'blackman', 'bartlett']:
        func = getattr(windows, window_name)

        direct_result = func(M, sym=True)
        get_window_result = windows.get_window(window_name, M, fftbins=True)

        assert np.allclose(direct_result, get_window_result, rtol=1e-10, atol=1e-10), \
            f"get_window('{window_name}', {M}) should match {window_name}({M})"
```

**Failing input**: `M=2` (and all other values)

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as windows

M = 10

direct = windows.hann(M)
via_get_window = windows.get_window('hann', M)

print(f"hann({M}):")
print(direct)

print(f"\nget_window('hann', {M}):")
print(via_get_window)

print(f"\nAre they equal? {np.allclose(direct, via_get_window)}")
```

**Output:**
```
hann(10):
[0.         0.11697778 0.41317591 0.75       0.96984631 0.96984631
 0.75       0.41317591 0.11697778 0.        ]

get_window('hann', 10):
[0.        0.0954915 0.3454915 0.6545085 0.9045085 1.        0.9045085
 0.6545085 0.3454915 0.0954915]

Are they equal? False
```

## Why This Is A Bug

This violates the principle of least surprise and creates an API inconsistency:

1. Users would reasonably expect `get_window('hann', M)` to be equivalent to `hann(M)`
2. The individual window functions (hann, hamming, blackman, etc.) default to `sym=True`, which creates symmetric windows for filter design
3. However, `get_window` defaults to `fftbins=True`, which internally calls the window functions with `sym=False` to create periodic windows
4. This means the "convenience" function `get_window` produces different results than directly calling the window functions with their default parameters

This is particularly problematic because:
- The documentation doesn't clearly warn users about this inconsistency
- Users migrating between direct calls and `get_window` will get different results
- The name "get_window" suggests it's a simple dispatcher, not that it changes the default behavior

## Fix

The fix should align the defaults. The most user-friendly approach would be to change `get_window`'s default to `fftbins=False` to match the individual functions' `sym=True` default:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -xxxx,7 +xxxx,7 @@
-def get_window(window, Nx, fftbins=True, *, xp=None, device=None):
+def get_window(window, Nx, fftbins=False, *, xp=None, device=None):
     """
     Return a window of a given length and type.
```

Alternatively, if the current default is intentional for FFT use cases, the documentation should prominently warn users:

```diff
+    .. warning::
+        By default, `get_window` uses `fftbins=True` which produces periodic windows.
+        This differs from calling window functions directly (e.g., `hann(Nx)`) which
+        default to symmetric windows (`sym=True`). To match direct calls, use
+        `get_window(window, Nx, fftbins=False)`.
```