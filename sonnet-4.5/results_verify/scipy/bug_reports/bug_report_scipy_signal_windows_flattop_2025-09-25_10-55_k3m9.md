# Bug Report: scipy.signal.windows.flattop Normalization Violation

**Target**: `scipy.signal.windows.flattop`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `flattop` window function violates its documented contract that "the maximum value normalized to 1". For all odd values of M, the function returns a maximum value of 1.000000003, exceeding the claimed normalization by 3×10⁻⁹.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as windows


@given(st.integers(min_value=1, max_value=1000))
@settings(max_examples=300)
def test_normalization_property(M):
    w = windows.flattop(M)
    max_val = np.max(w)

    assert max_val <= 1.0, f"flattop({M}) has max value {max_val} > 1.0"
```

**Failing input**: `M=3` (and all odd M values)

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as windows

for M in [3, 5, 7, 9, 11]:
    w = windows.flattop(M)
    max_val = np.max(w)
    print(f"flattop({M}): max = {max_val:.15f}")
```

Output:
```
flattop(3): max = 1.000000003000000
flattop(5): max = 1.000000003000000
flattop(7): max = 1.000000003000000
flattop(9): max = 1.000000003000000
flattop(11): max = 1.000000003000000
```

## Why This Is A Bug

The docstring explicitly states: "The window, with the maximum value normalized to 1". However, the function consistently returns values exceeding 1.0 by 3×10⁻⁹ for all odd M values.

The root cause is that the flattop coefficients sum to 1.000000003 instead of exactly 1.0:

```python
coeffs = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
sum(coeffs)  # = 1.000000003
```

Since the general cosine window reaches its maximum when all cosine terms align (at n=(M-1)/2 for odd M), the maximum equals the sum of coefficients.

## Fix

The coefficients should be normalized to sum to exactly 1.0:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -1234,7 +1234,7 @@ def flattop(M, sym=True, *, xp=None, device=None):
     """
     xp = _namespace(xp)
     a = xp.asarray(
-        [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368],
+        [0.21557894935326313, 0.41663157875010526, 0.27726315716821054, 0.08357894674926315, 0.006947367979157896],
         dtype=xp.float64, device=device
     )
     device = xp_device(a)
```

Alternatively, the implementation could normalize the coefficients at runtime, though this adds computational overhead.