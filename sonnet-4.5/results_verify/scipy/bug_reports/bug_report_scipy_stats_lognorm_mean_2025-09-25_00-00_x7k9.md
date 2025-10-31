# Bug Report: scipy.stats.lognorm Numerical Overflow in Mean Calculation

**Target**: `scipy.stats.lognorm.mean`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The lognormal distribution's mean calculation overflows to infinity for moderately large shape parameters (s ≥ 27), when it should return a finite (though large) value.

## Property-Based Test

```python
import numpy as np
import scipy.stats
import math
from hypothesis import given, strategies as st, settings

@given(st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False))
@settings(max_examples=500)
def test_lognorm_moments(s):
    """Lognormal mean should match theoretical value"""
    mean = scipy.stats.lognorm.mean(s)
    expected = np.exp(s ** 2 / 2)

    assert math.isclose(mean, expected, rel_tol=1e-9), \
        f"lognorm.mean(s={s}) = {mean}, expected {expected}"
```

**Failing input**: `s=27.0`

## Reproducing the Bug

```python
import numpy as np
import scipy.stats

s = 27.0

expected_mean = np.exp(s**2 / 2)
actual_mean = scipy.stats.lognorm.mean(s)

print(f"s = {s}")
print(f"Expected mean: {expected_mean}")
print(f"Actual mean: {actual_mean}")

assert not np.isinf(actual_mean), f"Mean should be finite, got {actual_mean}"
```

Output:
```
s = 27.0
Expected mean: 1.9968187854791924e+158
Actual mean: inf
AssertionError: Mean should be finite, got inf
```

## Why This Is A Bug

The lognormal distribution with shape parameter `s` has mean `exp(s²/2)`. For `s=27`, this equals `exp(364.5) ≈ 1.997e+158`, which is well within the range of float64 (max ≈ 1.8e+308).

However, scipy's implementation in `_continuous_distns.py` at line 6882 computes:
```python
p = np.exp(s*s)  # exp(s²) = exp(729) = inf (overflow)
mu = np.sqrt(p)  # sqrt(inf) = inf
```

This overflows because `exp(s²) = exp(729) ≈ 1.6e+316` exceeds float64's maximum value, even though the final result `sqrt(exp(s²)) = exp(s²/2)` would not overflow.

## Fix

```diff
--- a/scipy/stats/_continuous_distns.py
+++ b/scipy/stats/_continuous_distns.py
@@ -6879,8 +6879,8 @@ class lognorm_gen(rv_continuous):
         return np.where(x == 0, _XMIN, x)

     def _stats(self, s):
-        p = np.exp(s*s)
-        mu = np.sqrt(p)
+        p = np.exp(s*s / 2)
+        mu = p
         mu2 = p*(p-1)
         g1 = np.sqrt(p-1)*(2+p)
         g2 = np.polyval([1, 2, 3, 0, -6.0], p)
```

Wait, this would break mu2, g1, and g2 calculations. Let me reconsider. Actually, a better fix is:

```diff
--- a/scipy/stats/_continuous_distns.py
+++ b/scipy/stats/_continuous_distns.py
@@ -6879,7 +6879,7 @@ class lognorm_gen(rv_continuous):
         return np.where(x == 0, _XMIN, x)

     def _stats(self, s):
-        p = np.exp(s*s)
+        p = np.exp(np.minimum(s*s, 709))  # Prevent overflow; exp(709) ≈ 8.2e+307
         mu = np.sqrt(p)
         mu2 = p*(p-1)
         g1 = np.sqrt(p-1)*(2+p)
```

Actually, the best fix is to handle the mean calculation separately to avoid overflow:

```diff
--- a/scipy/stats/_continuous_distns.py
+++ b/scipy/stats/_continuous_distns.py
@@ -6879,8 +6879,14 @@ class lognorm_gen(rv_continuous):
         return np.where(x == 0, _XMIN, x)

     def _stats(self, s):
-        p = np.exp(s*s)
-        mu = np.sqrt(p)
+        # Compute mean directly to avoid overflow
+        # mu = sqrt(exp(s^2)) = exp(s^2/2)
+        mu = np.exp((s*s) / 2)
+
+        # For other moments, check for potential overflow
+        p_arg = s*s
+        p = np.where(p_arg < 709, np.exp(p_arg), np.inf)
+
         mu2 = p*(p-1)
         g1 = np.sqrt(p-1)*(2+p)
         g2 = np.polyval([1, 2, 3, 0, -6.0], p)
```