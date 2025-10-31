# Bug Report: scipy.signal.windows.flattop Maximum Value Exceeds Documented Limit

**Target**: `scipy.signal.windows.flattop`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `flattop` window function produces values that slightly exceed the documented maximum of 1.0, violating its API contract.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.signal import windows


window_functions_no_params = [
    'boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
    'blackmanharris', 'flattop', 'bartlett', 'barthann',
    'hamming', 'cosine', 'hann', 'lanczos', 'tukey'
]


@given(
    window_name=st.sampled_from(window_functions_no_params),
    M=st.integers(min_value=1, max_value=10000)
)
@settings(max_examples=500)
def test_normalization_property(window_name, M):
    window = windows.get_window(window_name, M, fftbins=True)
    max_val = np.max(np.abs(window))
    assert max_val <= 1.0 + 1e-10
```

**Failing input**: `window_name='flattop', M=2`

## Reproducing the Bug

```python
import numpy as np
from scipy.signal import windows

window = windows.flattop(3, sym=True)
max_val = np.max(window)

print(f"flattop(3, sym=True) = {window}")
print(f"max value = {max_val:.15f}")
print(f"exceeds 1.0? {max_val > 1.0}")

assert max_val <= 1.0, f"Maximum value {max_val} exceeds documented limit of 1.0"
```

Output:
```
flattop(3, sym=True) = [-4.21051e-04  1.00000e+00 -4.21051e-04]
max value = 1.000000003000000
exceeds 1.0? True
AssertionError: Maximum value 1.000000003 exceeds documented limit of 1.0
```

## Why This Is A Bug

The documentation for `flattop` explicitly states:

> Returns
> -------
> w : ndarray
>     The window, with the maximum value normalized to 1

However, the implementation produces values that exceed 1.0 (by approximately 3e-09) for certain window sizes.

**Root cause**: The flattop window is implemented as a sum of weighted cosine terms:
```python
a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
w = sum(a[k] * cos(k * fac))
```

At certain positions (e.g., the center for odd M with sym=True), all cosine terms equal 1.0, giving:
```python
w_center = sum(a[k]) = 1.000000003
```

The coefficients sum to slightly more than 1.0 due to rounding errors in the published values.

## Fix

Normalize the coefficients so they sum to exactly 1.0:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -668,7 +668,7 @@ def flattop(M, sym=True, *, xp=None, device=None):
     """
     xp = _namespace(xp)
     a = xp.asarray(
-        [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368],
+        [0.21557894935326313, 0.41663157875010526, 0.27726315716821054, 0.08357894674926315, 0.006947367979157896],
         dtype=xp.float64, device=device
     )
     device = xp_device(a)
```

These normalized coefficients sum to exactly 1.0 while preserving the original values to within 3e-09 (the same magnitude as the original error).