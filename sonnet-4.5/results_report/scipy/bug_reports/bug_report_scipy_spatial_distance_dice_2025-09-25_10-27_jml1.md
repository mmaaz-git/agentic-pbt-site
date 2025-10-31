# Bug Report: scipy.spatial.distance.dice Returns NaN for All-False Vectors

**Target**: `scipy.spatial.distance.dice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `dice` function returns NaN when comparing two all-False boolean vectors, violating the metric property that d(x, x) = 0. The similar `jaccard` function correctly handles this case and returns 0.

## Property-Based Test

```python
import numpy as np
import scipy.spatial.distance as distance
from hypothesis import given, strategies as st


@given(
    st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=50),
    st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=50)
)
def test_dice_bounds(u, v):
    assume(len(u) == len(v))
    u_arr = np.array(u, dtype=bool)
    v_arr = np.array(v, dtype=bool)

    d = distance.dice(u_arr, v_arr)

    assert 0 <= d <= 1, f"Dice dissimilarity should be in [0,1], got {d}"
```

**Failing input**: `u=[0, 0, 0, 0, 0]`, `v=[0, 0, 0, 0, 0]`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.distance import dice, jaccard

u = np.array([False, False, False, False, False])
v = np.array([False, False, False, False, False])

result = dice(u, v)
print(f"dice(all-False, all-False) = {result}")
print(f"Expected: 0.0")
print(f"For comparison, jaccard(all-False, all-False) = {jaccard(u, v)}")
```

Output:
```
dice(all-False, all-False) = nan
Expected: 0.0
For comparison, jaccard(all-False, all-False) = 0.0
```

## Why This Is A Bug

1. **Violates metric axiom**: For any dissimilarity/distance metric, d(x, x) = 0 must hold. The `dice` function returns NaN for identical all-False vectors.

2. **Inconsistent with similar function**: The `jaccard` function, which has a similar formula structure, was fixed for this exact issue in scipy 1.2.0 (see jaccard docstring: "Previously, if all... elements are zero, the function would return nan. This was changed to return 0 instead").

3. **Realistic input**: Comparing sparse boolean vectors (e.g., empty feature sets) is common in applications like document similarity and set comparison.

4. **Division by zero**: The formula (c_TF + c_FT) / (2*c_TT + c_TF + c_FT) produces 0/0 when both vectors are all-False, yielding NaN.

## Fix

Apply the same fix that was applied to `jaccard` in scipy 1.2.0:

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1500,7 +1500,9 @@ def dice(u, v, w=None):
             ntt = (u * v * w).sum()
     (nft, ntf) = _nbool_correspond_ft_tf(u, v, w=w)
-    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
+    denom = 2.0 * ntt + ntf + nft
+    return float((ntf + nft) / denom) if denom != 0 else 0.0
```

This ensures that when both vectors are all-False (or all-zero when weighted), the function returns 0.0 instead of NaN, consistent with the metric property d(x, x) = 0 and the behavior of `jaccard`.
