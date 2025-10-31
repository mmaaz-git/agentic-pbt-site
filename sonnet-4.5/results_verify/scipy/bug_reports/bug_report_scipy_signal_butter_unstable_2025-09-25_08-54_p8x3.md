# Bug Report: scipy.signal.butter Produces Unstable Filters

**Target**: `scipy.signal.butter`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

scipy.signal.butter produces unstable digital filters (with poles outside the unit circle) for certain valid parameter combinations, specifically high-order filters with very low cutoff frequencies. Butterworth filters are analytically guaranteed to be stable, so this represents a numerical precision bug.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.signal


@given(
    N=st.integers(min_value=1, max_value=10),
    Wn=st.floats(min_value=0.01, max_value=0.99)
)
@settings(max_examples=500)
def test_butter_returns_stable_filter(N, Wn):
    b, a = scipy.signal.butter(N, Wn)

    assert len(b) > 0 and len(a) > 0
    assert a[0] != 0

    roots_a = np.roots(a)
    max_magnitude = np.max(np.abs(roots_a))

    assert max_magnitude < 1.0 + 1e-6, \
        f"Butterworth filter should be stable (poles inside unit circle), but max pole magnitude = {max_magnitude}"
```

**Failing input**: `N=10, Wn=0.015625`

## Reproducing the Bug

```python
import numpy as np
import scipy.signal

N = 10
Wn = 0.015625

b, a = scipy.signal.butter(N, Wn)

poles = np.roots(a)
pole_magnitudes = np.abs(poles)
max_magnitude = np.max(pole_magnitudes)

print(f"Max pole magnitude: {max_magnitude:.10f}")
print(f"Is stable? {max_magnitude < 1.0}")
```

Output:
```
Max pole magnitude: 1.0064305588
Is stable? False
```

## Why This Is A Bug

Butterworth filters are **analytically guaranteed to be stable** - all poles should lie strictly inside the unit circle for digital filters. The butter() function is producing a filter with poles at magnitude 1.0064, which is outside the unit circle, making the filter unstable.

This occurs due to numerical precision issues when:
1. Filter order is high (N=10)
2. Cutoff frequency is very low (Wn=0.015625)

The combination of these factors causes numerical errors in the bilinear transformation or pole calculation that push poles outside the unit circle.

An unstable filter will produce unbounded output and cause severe problems in any signal processing application.

## Fix

The issue likely stems from numerical precision problems in the bilinear transformation. Possible fixes:

1. Use second-order sections (SOS) format instead of transfer function for high-order filters:
   ```python
   sos = scipy.signal.butter(N, Wn, output='sos')
   ```
   This is more numerically stable.

2. Add validation to warn users when filters are close to instability

3. Improve the numerical precision of the bilinear transformation for extreme parameter values

A simple validation and warning:

```diff
diff --git a/scipy/signal/_filter_design.py b/scipy/signal/_filter_design.py
index 1234567..abcdefg 100644
--- a/scipy/signal/_filter_design.py
+++ b/scipy/signal/_filter_design.py
@@ -2500,6 +2500,14 @@ def butter(N, Wn, btype='low', analog=False, output='ba', fs=None):
         b, a = zpk2tf(z, p, k)
     else:
         b, a = normalize(b, a)
+
+    if output == 'ba':
+        # Check filter stability
+        poles = np.roots(a)
+        if np.any(np.abs(poles) >= 1.0):
+            import warnings
+            warnings.warn(f"Designed filter may be unstable (max pole magnitude: {np.max(np.abs(poles)):.6f}). "
+                         "Consider using output='sos' for better numerical stability.")

     return b, a
```