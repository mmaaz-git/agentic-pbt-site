# Bug Report: scipy.spatial.distance.correlation Returns NaN for Constant Arrays

**Target**: `scipy.spatial.distance.correlation`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `correlation` distance function returns `nan` when given constant arrays (zero variance), violating the metric property that the distance between identical vectors should be 0.

## Property-Based Test

```python
from scipy.spatial.distance import correlation
from hypothesis import given, strategies as st
import numpy as np


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=2, max_size=20)
)
def test_correlation_identity(u):
    u_arr = np.array(u)
    d = correlation(u_arr, u_arr)
    assert np.isclose(d, 0.0, atol=1e-9)
```

**Failing input**: `u=[5.0, 5.0, 5.0]` (or any constant array)

## Reproducing the Bug

```python
from scipy.spatial.distance import correlation
import numpy as np

u = np.array([5.0, 5.0, 5.0])
v = np.array([5.0, 5.0, 5.0])

result = correlation(u, v)
print(f"correlation([5, 5, 5], [5, 5, 5]) = {result}")

assert result == 0.0, f"Expected 0.0 for identical arrays, got {result}"
```

## Why This Is A Bug

The correlation distance should satisfy d(x, x) = 0 for any vector x. When both inputs are constant arrays, they have zero variance. After mean-centering, all values become 0, making the denominator sqrt(uu * vv) equal to 0, which causes division by zero producing `nan`. For identical constant arrays, the distance should be 0.

## Fix

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