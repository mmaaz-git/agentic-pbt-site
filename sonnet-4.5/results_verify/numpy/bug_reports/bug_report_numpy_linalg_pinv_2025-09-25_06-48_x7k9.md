# Bug Report: numpy.linalg.pinv Subnormal Value Overflow

**Target**: `numpy.linalg.pinv`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.linalg.pinv` produces NaN and inf values when given matrices containing subnormal (denormalized) floating point numbers, violating the documented property `a @ pinv(a) @ a == a`.

## Property-Based Test

```python
import numpy as np
import numpy.linalg as la
from hypothesis import given, strategies as st, settings

def matrices(min_size=1, max_size=5):
    n = st.integers(min_value=min_size, max_value=max_size)
    return n.flatmap(lambda size: st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=size*size,
        max_size=size*size
    ).map(lambda vals: np.array(vals).reshape(size, size)))

@settings(max_examples=300)
@given(matrices(min_size=2, max_size=5))
def test_pinv_reconstruction(a):
    pinv_a = la.pinv(a)
    reconstructed = a @ pinv_a @ a
    assert np.allclose(reconstructed, a, rtol=1e-4, atol=1e-7)
```

**Failing input**: Matrix with subnormal value 2.2250738585e-313:
```python
a = np.array([[0.0, 0.0],
              [0.0, 2.2250738585e-313]])
```

## Reproducing the Bug

```python
import numpy as np
import numpy.linalg as la

a = np.array([[0.0, 0.0],
              [0.0, 2.2250738585e-313]])

pinv_a = la.pinv(a)
print("pinv(a):")
print(pinv_a)

reconstructed = a @ pinv_a @ a
print("\na @ pinv(a) @ a:")
print(reconstructed)
print("\nExpected:")
print(a)
print("\nContains NaN:", np.any(np.isnan(reconstructed)))
```

Output:
```
pinv(a):
[[nan nan]
 [nan inf]]

a @ pinv(a) @ a:
[[nan nan]
 [nan nan]]

Expected:
[[0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 2.22507386e-313]]

Contains NaN: True
```

## Why This Is A Bug

The documentation for `pinv` explicitly states that the property `a @ pinv(a) @ a == a` should hold (shown in the Examples section). However, when the input matrix contains subnormal floating point values (values smaller than the smallest normal float ~2.22e-308), the function produces NaN and inf values, violating this fundamental property.

The issue occurs because:
1. SVD computes a singular value of 2.22e-313
2. `pinv` attempts to compute the reciprocal: `1 / 2.22e-313 ≈ 4.49e312`
3. This exceeds the maximum representable float (~1.8e308), causing overflow to inf
4. The inf value propagates through matrix multiplication, producing NaN

While subnormal values are extreme edge cases, they are valid floating point values that numpy should handle gracefully rather than producing NaN.

## Fix

The fix should check for potential overflow when computing reciprocals of singular values. Instead of allowing overflow to inf, the code should either:
1. Set the reciprocal to zero (treating it as a small singular value)
2. Clamp the reciprocal to a maximum value
3. Use higher precision arithmetic for the reciprocal computation

A simple fix would be to add overflow checking:

```diff
--- a/numpy/linalg/_linalg.py
+++ b/numpy/linalg/_linalg.py
@@ -2282,7 +2282,11 @@ def pinv(a, rcond=None, hermitian=False, *, rtol=None):
     cutoff = rcond[..., nxp.newaxis] * s[..., 0:1]
     large = s > cutoff
-    s = divide(1, s, where=large, out=s)
+    with np.errstate(over='ignore', invalid='ignore'):
+        s_inv = divide(1, s, where=large, out=s)
+        overflow_mask = ~np.isfinite(s_inv)
+        s_inv[overflow_mask] = 0
+        s = s_inv
     s[~large] = 0
```

Note: This is a conceptual fix. The actual implementation may need adjustments based on numpy's internal structure.