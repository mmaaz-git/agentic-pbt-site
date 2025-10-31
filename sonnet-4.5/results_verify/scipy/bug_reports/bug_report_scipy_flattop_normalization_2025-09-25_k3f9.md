# Bug Report: scipy.signal.windows.flattop Normalization Exceeds 1.0

**Target**: `scipy.signal.windows.flattop`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `flattop` window function claims in its documentation to return "a window, with the maximum value normalized to 1", but for all odd values of M, the actual maximum value is 1.000000003, exceeding 1.0 by approximately 3e-9.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as windows


@given(st.integers(min_value=3, max_value=100).filter(lambda x: x % 2 == 1))
@settings(max_examples=50)
def test_flattop_normalization_bug(M):
    """
    The flattop window claims to return a window "with the maximum value
    normalized to 1", but for odd M values, the maximum value exceeds 1.0.
    """
    w = windows.flattop(M)
    max_val = np.max(w)

    assert max_val <= 1.0, \
        f"flattop({M}) has max {max_val:.15f} > 1.0 (exceeds by {max_val - 1.0:.3e})"
```

**Failing input**: `M=3` (or any odd integer >= 3)

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as windows

w = windows.flattop(3)
print(f"Window values: {w}")
print(f"Maximum value: {np.max(w):.15f}")
print(f"Expected: <= 1.0")
print(f"Exceeds 1.0 by: {np.max(w) - 1.0:.3e}")
```

Output:
```
Window values: [-4.21051e-04  1.00000e+00 -4.21051e-04]
Maximum value: 1.000000003000000
Expected: <= 1.0
Exceeds 1.0 by: 3.000e-09
```

## Why This Is A Bug

The function's docstring explicitly states:

> Returns
> -------
> w : ndarray
>     The window, with the maximum value normalized to 1

However, for all odd M values, the maximum value is 1.000000003, not 1.0. This violates the documented contract.

**Root cause**: The flattop window coefficients are defined as:
```python
a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
```

These coefficients sum to 1.000000003 instead of exactly 1.0. For odd M values, the window is evaluated at the center point where all cosine terms equal 1, resulting in the sum of coefficients becoming the maximum value.

Other cosine-based windows (hamming, hann, blackman, blackmanharris, nuttall) do not have this issue because their coefficients are more precisely specified.

## Fix

The coefficients should be adjusted to sum to exactly 1.0. Here's a patch:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -545,7 +545,7 @@ def flattop(M, sym=True, *, xp=None, device=None):
     """
     xp = _namespace(xp)
     a = xp.asarray(
-        [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368],
+        [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947365],
         dtype=xp.float64, device=device
     )
     device = xp_device(a)
```

The last coefficient is changed from `0.006947368` to `0.006947365` so that the sum equals exactly 1.0:
- Old sum: 0.21557895 + 0.41663158 + 0.277263158 + 0.083578947 + 0.006947368 = 1.000000003
- New sum: 0.21557895 + 0.41663158 + 0.277263158 + 0.083578947 + 0.006947365 = 1.000000000

Alternatively, all coefficients could be recomputed with higher precision to ensure the exact normalization while maintaining the window's spectral properties.