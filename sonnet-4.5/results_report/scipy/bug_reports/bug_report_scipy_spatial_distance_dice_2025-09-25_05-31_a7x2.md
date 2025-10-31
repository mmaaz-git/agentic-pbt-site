# Bug Report: scipy.spatial.distance.dice NaN on All-False Inputs

**Target**: `scipy.spatial.distance.dice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `dice` dissimilarity function returns NaN instead of 0 when both input boolean vectors contain only False values, violating the fundamental distance metric property that d(x, x) = 0.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.spatial.distance import dice


@given(
    st.integers(min_value=2, max_value=30),
    st.integers(min_value=0, max_value=1000)
)
@settings(max_examples=300)
def test_dice_identical_is_zero(n, seed):
    np.random.seed(seed)
    u = np.random.randint(0, 2, n, dtype=bool)

    d = dice(u, u)

    assert not np.isnan(d), \
        f"Dice distance should not return NaN"
    assert np.allclose(d, 0.0, rtol=1e-10, atol=1e-10), \
        f"Dice distance of identical vectors should be 0"
```

**Failing input**: `u = np.array([False, False]), v = np.array([False, False])`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.distance import dice

u = np.array([False, False])
v = np.array([False, False])

result = dice(u, v)
print(f"dice([False, False], [False, False]) = {result}")
print(f"Expected: 0.0")
print(f"Is NaN: {np.isnan(result)}")
```

Output:
```
dice([False, False], [False, False]) = nan
Expected: 0.0
Is NaN: True
```

## Why This Is A Bug

The Dice dissimilarity is defined as `(c_TF + c_FT) / (2*c_TT + c_FT + c_TF)` where c_ij counts occurrences.

When both vectors are all False:
- c_TT = 0 (no True & True pairs)
- c_TF = 0 (no True in u & False in v pairs)
- c_FT = 0 (no False in u & True in v pairs)
- Denominator = 2*0 + 0 + 0 = 0
- Result = 0/0 = NaN

This violates the fundamental property that d(x, x) = 0 for any distance metric. For identical vectors, dissimilarity should always be 0.

## Fix

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1500,7 +1500,10 @@ def dice(u, v, w=None):
             ntt = (u * v * w).sum()
     (nft, ntf) = _nbool_correspond_ft_tf(u, v, w=w)
-    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
+    denominator = 2.0 * ntt + ntf + nft
+    if denominator == 0:
+        return 0.0
+    return float((ntf + nft) / np.array(denominator))
```