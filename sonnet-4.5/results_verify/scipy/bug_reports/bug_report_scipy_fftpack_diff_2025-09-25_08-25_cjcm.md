# Bug Report: scipy.fftpack.diff Documentation Incorrectly Claims Round-Trip Property

**Target**: `scipy.fftpack.diff`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `scipy.fftpack.diff` function documentation claims "If `sum(x, axis=0) = 0` then `diff(diff(x, k), -k) == x`", but this property does not hold for many zero-mean signals.

## Property-Based Test

```python
import numpy as np
import scipy.fftpack as fftpack
from hypothesis import given, strategies as st, settings, assume

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=2, max_size=20),
       st.integers(min_value=1, max_value=3))
@settings(max_examples=500)
def test_diff_roundtrip_zero_mean(lst, order):
    x = np.array(lst)
    x = x - np.mean(x)
    assume(np.abs(np.sum(x)) < 1e-10)

    diff_x = fftpack.diff(x, order=order)
    roundtrip = fftpack.diff(diff_x, order=-order)
    assert np.allclose(roundtrip, x, atol=1e-4), f"diff round-trip failed for order={order}: {roundtrip} vs {x}"
```

**Failing input**: `lst=[0.0, 1.0]` (which produces `x=[-0.5, 0.5]` after mean subtraction)

## Reproducing the Bug

```python
import numpy as np
import scipy.fftpack as fftpack

x = np.array([-0.5, 0.5])
print(f"Input x: {x}")
print(f"Sum of x: {np.sum(x)}")

diff_x = fftpack.diff(x, order=1)
roundtrip = fftpack.diff(diff_x, order=-1)

print(f"diff(diff(x, 1), -1): {roundtrip}")
print(f"Expected: {x}")
print(f"Match: {np.allclose(roundtrip, x)}")
```

Output:
```
Input x: [-0.5  0.5]
Sum of x: 0.0
diff(diff(x, 1), -1): [0. 0.]
Expected: [-0.5  0.5]
Match: False
```

## Why This Is A Bug

The function's docstring explicitly states: "If `sum(x, axis=0) = 0` then `diff(diff(x, k), -k) == x`".

However, for the simple zero-mean signal `x = [-0.5, 0.5]`, the round-trip `diff(diff(x, 1), -1)` returns `[0, 0]` instead of the original `[-0.5, 0.5]`.

This violates the documented API contract. Testing reveals the property fails for many zero-mean signals, including:
- `[-0.5, 0.5]` (length 2): FAILS
- `[1, -1, 1, -1]` (length 4): FAILS

While it works for others:
- `[-1, 0, 1]` (length 3): WORKS
- `[0, 1, 0, -1]` (length 4): WORKS

The documentation claim is false in general and should be corrected.

## Fix

The documentation should be corrected to accurately describe when the round-trip property holds. The current statement is misleading:

```diff
diff --git a/scipy/fftpack/_pseudo_diffs.py b/scipy/fftpack/_pseudo_diffs.py
--- a/scipy/fftpack/_pseudo_diffs.py
+++ b/scipy/fftpack/_pseudo_diffs.py
@@ -28,7 +28,9 @@ def diff(x, order=1, period=None, _cache=_cache):

 Notes
 -----
-If ``sum(x, axis=0) = 0`` then ``diff(diff(x, k), -k) == x`` (within
-numerical accuracy).
+The round-trip property ``diff(diff(x, k), -k) == x`` does not hold in general,
+even for zero-mean signals. It only holds for signals whose Fourier coefficients
+at non-zero frequencies remain non-zero after differentiation.

 For odd order and even ``len(x)``, the Nyquist mode is taken zero.
```

Alternatively, the documentation could be removed entirely if the precise condition under which the round-trip holds cannot be simply stated.