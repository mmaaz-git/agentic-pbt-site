# Bug Report: scipy.spatial.distance.dice NaN on All-False Arrays

**Target**: `scipy.spatial.distance.dice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `dice` distance function returns NaN when both input boolean arrays contain only False values, due to division by zero in the Dice formula. This violates the expected behavior that identical inputs should have distance 0.

## Property-Based Test

```python
import math

import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst
from scipy.spatial import distance


@given(
    u=npst.arrays(
        dtype=np.bool_,
        shape=st.integers(min_value=1, max_value=100),
    ),
    v=npst.arrays(
        dtype=np.bool_,
        shape=st.integers(min_value=1, max_value=100),
    ),
)
@settings(max_examples=300)
def test_dice_symmetry(u, v):
    if u.shape != v.shape:
        return

    d_uv = distance.dice(u, v)
    d_vu = distance.dice(v, u)

    assert math.isclose(d_uv, d_vu, rel_tol=1e-9, abs_tol=1e-9)
```

**Failing input**: `u=array([False]), v=array([False])`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial import distance

u = np.array([False])
v = np.array([False])

result = distance.dice(u, v)
print(f"dice([False], [False]) = {result}")

print(f"\nFor comparison:")
print(f"jaccard([False], [False]) = {distance.jaccard(u, v)}")
```

Output:
```
dice([False], [False]) = nan
RuntimeWarning: invalid value encountered in divide

For comparison:
jaccard([False], [False]) = 0.0
```

## Why This Is A Bug

1. **Violates distance metric properties**: For identical inputs, a distance metric should return 0. Arrays `[False]` and `[False]` are identical, so the distance should be 0.

2. **Inconsistent with similar metrics**: The `jaccard` distance function correctly handles this edge case by returning 0.0 for all-False arrays.

3. **NaN breaks symmetry property**: While `dice(u, v)` and `dice(v, u)` both return NaN, NaN != NaN in Python, so `d_uv == d_vu` is False, violating symmetry.

4. **Undocumented behavior**: The documentation doesn't mention that NaN can be returned, nor does it specify preconditions on the input.

5. **Formula edge case**: The Dice formula `(c_TF + c_FT) / (2*c_TT + c_TF + c_FT)` produces 0/0 when all values are False, since c_TT = c_TF = c_FT = 0.

## Fix

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1500,7 +1500,10 @@ def dice(u, v, w=None):
         nft = (u_ne_v & v).sum()
         ntf = (u_ne_v & u).sum()
         ntt = (u & v).sum()
-    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
+    denominator = 2.0 * ntt + ntf + nft
+    if denominator == 0:
+        return 0.0
+    return float((ntf + nft) / np.array(denominator))
```

This fix returns 0.0 when both arrays are all False (identical inputs), consistent with the expected behavior of a dissimilarity metric and matching the behavior of `jaccard`.