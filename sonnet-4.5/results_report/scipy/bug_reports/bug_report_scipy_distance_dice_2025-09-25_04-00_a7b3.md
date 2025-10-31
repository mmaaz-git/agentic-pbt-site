# Bug Report: scipy.spatial.distance.dice Returns NaN for All-False Arrays

**Target**: `scipy.spatial.distance.dice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `dice` dissimilarity function returns `nan` when both input boolean arrays contain only `False` values, violating the metric property that the distance between identical vectors should be 0.

## Property-Based Test

```python
from scipy.spatial.distance import dice
from hypothesis import given, strategies as st, assume
import numpy as np


@given(
    st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=30),
    st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=30)
)
def test_dice_distance_range(u, v):
    assume(len(u) == len(v))
    u_arr = np.array(u, dtype=bool)
    v_arr = np.array(v, dtype=bool)
    d = dice(u_arr, v_arr)
    assert 0.0 <= d <= 1.0 + 1e-9
```

**Failing input**: `u=[0], v=[0]` (or any all-False arrays)

## Reproducing the Bug

```python
from scipy.spatial.distance import dice
import numpy as np

u = np.array([False, False, False])
v = np.array([False, False, False])

result = dice(u, v)
print(f"dice([False, False, False], [False, False, False]) = {result}")

assert result == 0.0, f"Expected 0.0, got {result}"
```

## Why This Is A Bug

The Dice dissimilarity is a distance metric that should satisfy the identity of indiscernibles property: d(x, x) = 0. When both inputs are all-False arrays, the denominator (2*c_TT + c_FT + c_TF) becomes 0, causing division by zero that produces `nan`. Two identical vectors should always have distance 0.

## Fix

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1500,4 +1500,7 @@ def dice(u, v, w=None):
         else:
             ntt = (u * v * w).sum()
     (nft, ntf) = _nbool_correspond_ft_tf(u, v, w=w)
-    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
+    denom = 2.0 * ntt + ntf + nft
+    if denom == 0:
+        return 0.0
+    return float((ntf + nft) / np.array(denom))
```