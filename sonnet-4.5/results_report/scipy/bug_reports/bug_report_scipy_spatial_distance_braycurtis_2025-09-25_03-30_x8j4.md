# Bug Report: scipy.spatial.distance.braycurtis Division by Zero

**Target**: `scipy.spatial.distance.braycurtis`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `braycurtis` distance function returns `nan` when both input vectors are all zeros, due to division by zero, violating the identity property.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.spatial.distance import braycurtis


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
@settings(max_examples=500)
def test_braycurtis_identity(u_list):
    u = np.array(u_list)
    d = braycurtis(u, u)
    assert np.isclose(d, 0.0), f"braycurtis(u, u) should be 0, got {d}"
```

**Failing input**: `u_list=[0.0, 0.0, 0.0]`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.distance import braycurtis

u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])
result = braycurtis(u, v)
assert np.isnan(result)
```

## Why This Is A Bug

1. **Violates identity property**: For any vector `u`, `braycurtis(u, u)` should return 0, but returns `nan` for all-zero vectors.

2. **Violates reasonable expectations**: Two identical all-zero vectors should have zero distance. The mathematical formula gives 0/0, but the limiting behavior suggests 0.

3. **Inconsistent with other distance functions**: Functions like `jaccard` and `canberra` handle all-zero cases gracefully by returning 0.

## Fix

Add a zero-check before division at line 1245:

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1242,7 +1242,11 @@ def braycurtis(u, v, w=None):
         w = _validate_weights(w)
         l1_diff = w * l1_diff
         l1_sum = w * l1_sum
-    return l1_diff.sum() / l1_sum.sum()
+    denominator = l1_sum.sum()
+    if denominator == 0:
+        return 0.0
+    else:
+        return l1_diff.sum() / denominator
```