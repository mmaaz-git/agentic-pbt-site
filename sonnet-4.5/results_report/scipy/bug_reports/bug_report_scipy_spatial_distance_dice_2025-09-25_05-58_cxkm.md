# Bug Report: scipy.spatial.distance.dice Identity Property Violation

**Target**: `scipy.spatial.distance.dice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `dice` dissimilarity function returns `nan` instead of `0.0` when both input boolean arrays are all False, violating the fundamental identity property that `d(x, x) = 0` for all x.

## Property-Based Test

```python
import numpy as np
import scipy.spatial.distance as dist
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays


@given(arrays(np.bool_, (10,), elements=st.booleans()))
def test_dice_identity_property(u):
    d = dist.dice(u, u)
    assert not np.isnan(d), f"dice(u, u) should not be NaN, got {d} for u={u}"
```

**Failing input**: `array([False, False, False, False, False, False, False, False, False, False])`

## Reproducing the Bug

```python
import numpy as np
import scipy.spatial.distance as dist

u = np.array([False, False, False, False, False])
v = np.array([False, False, False, False, False])

result = dist.dice(u, v)

assert np.array_equal(u, v)
print(f"dice(u, u) = {result}")
```

Output: `dice(u, u) = nan`

## Why This Is A Bug

The Dice dissimilarity is defined as:

```
dice(u, v) = (c_TF + c_FT) / (2*c_TT + c_TF + c_FT)
```

When both arrays are all False:
- `c_TT = 0` (no positions where both are True)
- `c_TF = 0` (no positions where u=True, v=False)
- `c_FT = 0` (no positions where u=False, v=True)

This results in `0/0 = nan`.

However, all dissimilarity measures must satisfy the identity property: `d(x, x) = 0`. Even though the arrays represent "empty sets," they are identical, so the dissimilarity should be 0, not NaN. This is consistent with how related measures like Jaccard handle this edge case (returning 0.0).

## Fix

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1500,7 +1500,10 @@ def dice(u, v, w=None):
         else:
             ntt = (u * v * w).sum()
     (nft, ntf) = _nbool_correspond_ft_tf(u, v, w=w)
-    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
+    denominator = 2.0 * ntt + ntf + nft
+    if denominator == 0:
+        return 0.0
+    return float((ntf + nft) / denominator)
```