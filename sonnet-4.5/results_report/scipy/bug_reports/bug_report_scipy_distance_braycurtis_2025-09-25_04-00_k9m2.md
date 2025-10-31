# Bug Report: scipy.spatial.distance.braycurtis Returns NaN for All-Zero Arrays

**Target**: `scipy.spatial.distance.braycurtis`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `braycurtis` distance function returns `nan` when both input arrays contain only zeros, violating the metric property that the distance between identical vectors should be 0.

## Property-Based Test

```python
from scipy.spatial.distance import braycurtis
from hypothesis import given, strategies as st, assume
import numpy as np


@given(
    st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=30),
    st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=30)
)
def test_braycurtis_range_positive_inputs(u, v):
    assume(len(u) == len(v))
    u_arr = np.array(u)
    v_arr = np.array(v)
    d = braycurtis(u_arr, v_arr)
    assert 0.0 <= d <= 1.0 + 1e-9
```

**Failing input**: `u=[0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0]`

## Reproducing the Bug

```python
from scipy.spatial.distance import braycurtis
import numpy as np

u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])

result = braycurtis(u, v)
print(f"braycurtis([0, 0, 0], [0, 0, 0]) = {result}")

assert result == 0.0, f"Expected 0.0, got {result}"
```

## Why This Is A Bug

The Bray-Curtis distance should satisfy d(x, x) = 0 for any vector x. When both inputs are all-zero arrays, the denominator sum(|u_i + v_i|) becomes 0, causing division by zero that produces `nan`. The docstring incorrectly states the distance "is undefined if the inputs are of length zero", but the actual issue is when all values are zero, not when the length is zero.

## Fix

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1242,4 +1242,7 @@ def braycurtis(u, v, w=None):
         w = _validate_weights(w)
         l1_diff = w * l1_diff
         l1_sum = w * l1_sum
-    return l1_diff.sum() / l1_sum.sum()
+    denom = l1_sum.sum()
+    if denom == 0:
+        return 0.0
+    return l1_diff.sum() / denom
```