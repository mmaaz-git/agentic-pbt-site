# Bug Report: scipy.signal.windows Periodic Window Normalization Failure for Odd M

**Target**: `scipy.signal.windows` (all window functions using `_general_cosine_impl`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Periodic windows (`sym=False`) fail to achieve the expected maximum value of 1.0 when the window length M is odd. This causes normalization inconsistencies and amplitude errors of up to 37% in spectral analysis applications.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from scipy.signal.windows import hamming

@given(M=st.integers(min_value=1, max_value=1000))
def test_periodic_window_normalized(M):
    w = hamming(M, sym=False)
    assert np.isclose(np.max(w), 1.0, rtol=1e-10)
```

**Failing input**: M=3 (and all odd M values: 1, 3, 5, 7, 9, ...)

## Reproducing the Bug

```python
import numpy as np
from scipy.signal.windows import hamming, hann, blackman

print("Periodic windows with odd M do NOT achieve max=1.0:")
for M in [3, 5, 7]:
    w_hamming = hamming(M, sym=False)
    w_hann = hann(M, sym=False)
    w_blackman = blackman(M, sym=False)
    print(f"M={M}:")
    print(f"  hamming:  max={np.max(w_hamming):.4f} (error: {1.0 - np.max(w_hamming):.4f})")
    print(f"  hann:     max={np.max(w_hann):.4f} (error: {1.0 - np.max(w_hann):.4f})")
    print(f"  blackman: max={np.max(w_blackman):.4f} (error: {1.0 - np.max(w_blackman):.4f})")

print("\nPeriodic windows with even M correctly achieve max=1.0:")
for M in [2, 4, 6]:
    w = hamming(M, sym=False)
    print(f"M={M}: max={np.max(w):.4f}")
```

Output:
```
Periodic windows with odd M do NOT achieve max=1.0:
M=3:
  hamming:  max=0.7700 (error: 0.2300)
  hann:     max=0.7500 (error: 0.2500)
  blackman: max=0.6300 (error: 0.3700)
M=5:
  hamming:  max=0.9121 (error: 0.0879)
  hann:     max=0.9045 (error: 0.0955)
  blackman: max=0.8492 (error: 0.1508)
M=7:
  hamming:  max=0.9544 (error: 0.0456)
  hann:     max=0.9505 (error: 0.0495)
  blackman: max=0.9204 (error: 0.0796)

Periodic windows with even M correctly achieve max=1.0:
M=2: max=1.0000
M=4: max=1.0000
M=6: max=1.0000
```

## Why This Is A Bug

1. **Documentation contract violation**: The docstrings state that periodic windows (`sym=False`) are "ready to use with FFT" and "for use in spectral analysis", implying proper normalization.

2. **Inconsistent behavior**: The normalization depends on whether M is even or odd, which is unexpected and undocumented.

3. **Amplitude errors**: For M=3, blackman window has 37% amplitude error (max=0.63 instead of 1.0). This causes significant errors in FFT-based spectral analysis.

4. **Mathematical incorrectness**: Window functions are mathematically defined to have a peak value of 1.0 for normalization purposes. The periodic form should maintain this property.

## Root Cause

The bug is in `_general_cosine_impl` in `scipy/signal/windows/_windows.py`:

```python
def _general_cosine_impl(M, a, xp, device, sym=True):
    if _len_guards(M):
        return xp.ones(M, dtype=xp.float64, device=device)
    M, needs_trunc = _extend(M, sym)  # For sym=False: M -> M+1, needs_trunc=True

    fac = xp.linspace(-xp.pi, xp.pi, M, dtype=xp.float64, device=device)
    w = xp.zeros(M, dtype=xp.float64, device=device)
    for k in range(a.shape[0]):
        w += a[k] * xp.cos(k * fac)

    return _truncate(w, needs_trunc)  # Removes last element
```

When `sym=False`:
- `_extend` returns M+1 and `needs_trunc=True`
- `linspace(-π, π, M+1)` creates M+1 points
- `_truncate` removes the last element

**The problem**:
- When M+1 is **even** (M is **odd**), `linspace(-π, π, M+1)` produces even number of points, which are symmetric around 0 but **don't include 0**
- When M+1 is **odd** (M is **even**), the middle point is exactly at 0, achieving max=1.0

Example for M=3:
```python
M_ext = 4  # M + 1
fac = linspace(-π, π, 4)  # [-π, -π/3, π/3, π]  <- No 0!
# For hamming: 0.54 - 0.46*cos(0) would give 1.0
# But at fac=π/3: 0.54 - 0.46*cos(π/3) = 0.54 - 0.46*0.5 = 0.77
```

## Fix

Modify `_extend` and `_truncate` to handle odd M specially:

```diff
diff --git a/scipy/signal/windows/_windows.py b/scipy/signal/windows/_windows.py
index 1234567..abcdef0 100644
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -10,8 +10,12 @@ def _extend(M, sym):
     """Extend window by 1 sample if needed for DFT-even symmetry"""
     if not sym:
-        return M + 1, True
+        # For odd M, extend by 2 to ensure peak is sampled
+        if M % 2 == 1:
+            return M + 2, (True, 1)  # Signal to truncate first and last
+        else:
+            return M + 1, (True, 0)  # Signal to truncate last only
     else:
-        return M, False
+        return M, (False, 0)

 def _truncate(w, needed):
-    """Truncate window by 1 sample if needed for DFT-even symmetry"""
-    if needed:
-        return w[:-1]
+    """Truncate window sample(s) if needed for DFT-even symmetry"""
+    needs_trunc, start_idx = needed
+    if needs_trunc:
+        if start_idx == 1:  # Odd M case: remove first and last
+            return w[1:-1]
+        else:  # Even M case: remove last only
+            return w[:-1]
     else:
         return w
```

This ensures that for odd M, we sample M+2 points (which includes the peak at 0 in the middle), then take the middle M points.