# Bug Report: scipy.spatial.distance.cosine Returns NaN for Zero Vectors

**Target**: `scipy.spatial.distance.cosine`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cosine` distance function returns `nan` when both input vectors are zero vectors, violating the metric property that the distance between identical vectors should be 0.

## Property-Based Test

```python
from scipy.spatial.distance import cosine
from hypothesis import given, strategies as st, assume
import numpy as np


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=20),
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=20)
)
def test_cosine_distance_range(u, v):
    assume(len(u) == len(v))
    u_arr = np.array(u)
    v_arr = np.array(v)
    d = cosine(u_arr, v_arr)
    assert 0.0 <= d <= 2.0 + 1e-9
```

**Failing input**: `u=[0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0]`

## Reproducing the Bug

```python
from scipy.spatial.distance import cosine
import numpy as np

u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])

result = cosine(u, v)
print(f"cosine([0, 0, 0], [0, 0, 0]) = {result}")

assert result == 0.0, f"Expected 0.0 for identical vectors, got {result}"
```

## Why This Is A Bug

The cosine distance should satisfy d(x, x) = 0 for any vector x. When both inputs are zero vectors, the function divides by the product of vector norms (both 0), resulting in `nan`. Two identical zero vectors should have cosine distance 0.

Note: `cosine` is implemented as `correlation(u, v, w=w, centered=False)`, so this bug stems from the same division-by-zero issue in the `correlation` function.

## Fix

Since `cosine` delegates to `correlation`, the fix should be applied in the `correlation` function:

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -679,5 +679,8 @@ def correlation(u, v, w=None, centered=True):
     uv = np.dot(u, vw)
     uu = np.dot(u, uw)
     vv = np.dot(v, vw)
-    dist = 1.0 - uv / math.sqrt(uu * vv)
+    denom = math.sqrt(uu * vv)
+    if denom == 0:
+        return 0.0 if np.allclose(u, v) else np.nan
+    dist = 1.0 - uv / denom
     return np.clip(dist, 0.0, 2.0)
```