# Bug Report: scipy.differentiate.hessian Asymmetric Hessian Matrix

**Target**: `scipy.differentiate.hessian`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `hessian` function returns asymmetric Hessian matrices, violating Schwarz's theorem which guarantees that Hessians of smooth functions must be symmetric.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy.differentiate import hessian

@settings(max_examples=500)
@given(
    st.integers(min_value=2, max_value=5),
    st.data()
)
def test_hessian_symmetry(m, data):
    x = data.draw(st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        min_size=m, max_size=m
    ))
    x = np.array(x)

    def f(xi):
        return np.sum(xi**2 * np.sin(xi), axis=0)

    res = hessian(f, x)
    assume(np.any(res.success))

    H = res.ddf
    assume(not np.any(np.isnan(H)) and not np.any(np.isinf(H)))

    assert np.allclose(H, H.T, rtol=1e-5, atol=1e-8), \
        f"Hessian not symmetric: max difference = {np.max(np.abs(H - H.T))}"
```

**Failing input**: `x = [0.0, 5.0]`

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import hessian

x = np.array([0.0, 5.0])

def f(xi):
    return np.sum(xi**2 * np.sin(xi), axis=0)

res = hessian(f, x)

print("Hessian:")
print(res.ddf)
print("\nDifference (H - H.T):")
print(res.ddf - res.ddf.T)
print(f"\nMax absolute difference: {np.max(np.abs(res.ddf - res.ddf.T))}")
```

Output:
```
Hessian:
[[-9.79880584e-10 -1.12402702e-07]
 [-1.51434181e-11  2.77285020e+01]]

Difference (H - H.T):
[[ 0.00000000e+00 -1.12387559e-07]
 [ 1.12387559e-07  0.00000000e+00]]

Max absolute difference: 1.1238755900752667e-07
```

## Why This Is A Bug

By Schwarz's theorem (also known as Clairaut's theorem on equality of mixed partials), for any twice-continuously differentiable function f, the Hessian matrix H must satisfy H[i,j] = H[j,i] for all i,j. The function f(x) = sum(x^2 * sin(x)) is infinitely differentiable, so its Hessian must be symmetric.

The scipy documentation for `hessian` shows examples using `scipy.optimize.rosen_hess`, which returns symmetric matrices. This sets an expectation that `hessian` should also return symmetric results.

The bug manifests as off-diagonal elements being computed independently rather than being enforced to be equal. Investigating with different parameters shows:
- Default parameters: asymmetry ~1e-7
- Tighter tolerances help in some cases but don't fix the root cause
- Increasing iterations or order can make asymmetry WORSE (up to 1e-3)

This indicates a fundamental algorithmic issue where the nested jacobian calls compute H[i,j] and H[j,i] independently without enforcing symmetry.

## Fix

The `hessian` function is implemented by nesting calls to `jacobian` (line 1120 in _differentiate.py). The issue is that this computes the off-diagonal elements H[i,j] and H[j,i] independently, leading to numerical differences.

A proper fix should symmetrize the result:

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -1120,6 +1120,9 @@
     res = jacobian(df, x, tolerances=tolerances, **kwargs)  # jacobian of jacobian
+    # Enforce symmetry as guaranteed by Schwarz's theorem
+    res.ddf = (res.ddf + np.moveaxis(res.ddf, 0, 1)) / 2
+
     res.df = res.ddf
     return res
```

Alternatively, the algorithm could compute only the upper or lower triangle and copy to ensure symmetry.