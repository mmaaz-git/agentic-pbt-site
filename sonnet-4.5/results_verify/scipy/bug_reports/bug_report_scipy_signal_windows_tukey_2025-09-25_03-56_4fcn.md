# Bug Report: scipy.signal.windows.tukey Produces NaN with Very Small Alpha

**Target**: `scipy.signal.windows.tukey`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `tukey` window function produces NaN values when the `alpha` parameter is very small but non-zero (approximately 1e-313 to 1e-310), due to numerical overflow in the computation `2.0/alpha`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as windows
import numpy as np

@settings(max_examples=300)
@given(
    M=st.integers(min_value=3, max_value=1000),
    alpha=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)
)
def test_tukey_no_nan(M, alpha):
    result = windows.tukey(M, alpha=alpha, sym=True)
    assert not np.any(np.isnan(result)), f"tukey({M}, alpha={alpha}) contains NaN"
    assert not np.any(np.isinf(result)), f"tukey({M}, alpha={alpha}) contains inf"
```

**Failing input**: `M=3, alpha=2.2250738585e-313` (and other very small alpha values)

## Reproducing the Bug

```python
import scipy.signal.windows as w
import numpy as np

result = w.tukey(3, alpha=1e-313, sym=True)
print(f'tukey(3, alpha=1e-313) = {result}')
print(f'Contains NaN: {np.any(np.isnan(result))}')

for alpha in [1e-320, 1e-313, 1e-310, 1e-300, 1e-10]:
    result = w.tukey(5, alpha=alpha)
    has_nan = np.any(np.isnan(result))
    print(f'alpha={alpha:.2e}: has_nan={has_nan}, result={result}')
```

Output:
```
tukey(3, alpha=1e-313) = [ 0.  1. nan]
Contains NaN: True
alpha=1.00e-320: has_nan=True, result=[ 0.  1.  1.  1. nan]
alpha=1.00e-313: has_nan=True, result=[ 0.  1.  1.  1. nan]
alpha=1.00e-310: has_nan=True, result=[ 0.  1.  1.  1. nan]
alpha=1.00e-300: has_nan=False, result=[0. 1. 1. 1. 1.]
alpha=1.00e-10: has_nan=False, result=[0. 1. 1. 1. 0.]
```

## Why This Is A Bug

The tukey window should return valid floating-point values for all valid input parameters. The parameter `alpha` is documented as a float in the range [0, 1], and very small positive values are valid inputs.

The root cause is numerical overflow in this line of the implementation:

```python
w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
```

When `alpha` is very small (e.g., 1e-313), the division `2.0/alpha` produces a value larger than the maximum representable float64 (approximately 1.8e308), causing overflow to infinity. When infinity is passed to `cos()`, and then added, the result is NaN.

The code already handles `alpha <= 0` as a special case (returning a boxcar window), but it doesn't handle extremely small positive values that cause numerical overflow.

## Fix

Add a check for very small alpha values and treat them the same as `alpha=0`:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -tukey function
     if _len_guards(M):
         return xp.ones(M, dtype=xp.float64, device=device)

-    if alpha <= 0:
+    # Treat very small alpha as 0 to avoid numerical overflow in division
+    if alpha <= 0 or alpha < 1e-300:
         return xp.ones(M, dtype=xp.float64, device=device)
     elif alpha >= 1.0:
         return hann(M, sym=sym, xp=xp, device=device)
```

Alternatively, use a more robust computation that avoids division by very small alpha:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -tukey function computation
     w1 = 0.5 * (1 + xp.cos(xp.pi * (-1 + 2.0*n1/alpha/(M-1))))
     w2 = xp.ones(n2.shape, device=device)
-    w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
+    # Rewrite to avoid division by alpha in a way that can overflow
+    w3 = 0.5 * (1 + xp.cos(xp.pi * (1 - 2.0*(1 - n3/(M-1))/alpha)))
```

However, the first fix (treating very small alpha as 0) is simpler and more robust.