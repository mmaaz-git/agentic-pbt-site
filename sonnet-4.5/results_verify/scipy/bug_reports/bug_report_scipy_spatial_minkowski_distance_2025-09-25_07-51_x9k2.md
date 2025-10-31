# Bug Report: scipy.spatial.minkowski_distance Numerical Underflow

**Target**: `scipy.spatial.minkowski_distance`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.spatial.minkowski_distance` returns 0.0 for very small input values due to numerical underflow, while the mathematically equivalent `scipy.spatial.distance.euclidean` function correctly returns the non-zero distance using a more stable BLAS implementation.

## Property-Based Test

```python
import numpy as np
import scipy.spatial
import scipy.spatial.distance as dist
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

@given(
    x=arrays(dtype=np.float64, shape=(5,), elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)),
    y=arrays(dtype=np.float64, shape=(5,), elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
)
@settings(max_examples=1000)
def test_minkowski_distance_consistency_with_euclidean(x, y):
    d_minkowski = scipy.spatial.minkowski_distance(x, y, p=2)
    d_euclidean = dist.euclidean(x, y)
    assert np.isclose(d_minkowski, d_euclidean, rtol=1e-10, atol=0)
```

**Failing input**: `x=array([0., 0., 0., 0., 0.])`, `y=array([2.22507386e-308, 2.22507386e-308, 2.22507386e-308, 2.22507386e-308, 2.22507386e-308])`

## Reproducing the Bug

```python
import numpy as np
import scipy.spatial
import scipy.spatial.distance as dist

x = np.array([0., 0., 0., 0., 0.])
y = np.array([2.22507386e-308, 2.22507386e-308, 2.22507386e-308, 2.22507386e-308, 2.22507386e-308])

d_minkowski = scipy.spatial.minkowski_distance(x, y, p=2)
d_euclidean = dist.euclidean(x, y)

print(f"scipy.spatial.minkowski_distance(x, y, p=2) = {d_minkowski}")
print(f"scipy.spatial.distance.euclidean(x, y) = {d_euclidean}")
```

Output:
```
scipy.spatial.minkowski_distance(x, y, p=2) = 0.0
scipy.spatial.distance.euclidean(x, y) = 4.975416402579851e-308
```

## Why This Is A Bug

The docstring for `minkowski_distance` states it computes the L**p distance, and for p=2 this should match the Euclidean distance. However, `minkowski_distance` uses a naive implementation (`np.sum(np.abs(y-x)**p)**(1/p)`) that suffers from underflow when squaring very small values near the float64 minimum (2.22e-308). The squared values underflow to 0.0, causing the function to incorrectly return 0.0.

In contrast, `scipy.spatial.distance.euclidean` delegates to `scipy.linalg.norm`, which uses BLAS's `nrm2` function. This function employs a numerically stable algorithm that avoids underflow and correctly computes the distance as approximately 4.98e-308.

## Fix

```diff
--- a/scipy/spatial/_kdtree.py
+++ b/scipy/spatial/_kdtree.py
@@ -51,8 +51,11 @@ def minkowski_distance_p(x, y, p=2):
     if p == np.inf:
         return np.amax(np.abs(y-x), axis=-1)
     elif p == 1:
         return np.sum(np.abs(y-x), axis=-1)
+    elif p == 2:
+        from scipy.linalg import norm
+        return norm(y-x, ord=2, axis=-1)**2
     else:
         return np.sum(np.abs(y-x)**p, axis=-1)
```

Alternatively, use `scipy.linalg.norm` throughout `minkowski_distance` for better numerical stability:

```diff
--- a/scipy/spatial/_kdtree.py
+++ b/scipy/spatial/_kdtree.py
@@ -77,8 +77,9 @@ def minkowski_distance(x, y, p=2):
     """
     x = np.asarray(x)
     y = np.asarray(y)
-    if p == np.inf or p == 1:
-        return minkowski_distance_p(x, y, p)
+    if p == 2:
+        from scipy.linalg import norm
+        return norm(y-x, ord=2, axis=-1)
     else:
-        return minkowski_distance_p(x, y, p)**(1./p)
+        return minkowski_distance_p(x, y, p)**(1./p)
```