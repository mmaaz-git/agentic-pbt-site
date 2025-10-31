# Bug Report: scipy.spatial.distance.dice Returns NaN for All-False Boolean Arrays

**Target**: `scipy.spatial.distance.dice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `dice` function returns NaN when both input boolean arrays contain only False values, instead of returning 0.0 for identical inputs.

## Property-Based Test

```python
import math
import numpy as np
from hypothesis import given, settings, strategies as st
from scipy.spatial import distance

@settings(max_examples=1000)
@given(n=st.integers(1, 20))
def test_dice_all_zeros(n):
    x = np.zeros(n, dtype=bool)
    y = np.zeros(n, dtype=bool)

    dist = distance.dice(x, y)

    assert math.isclose(dist, 0.0, abs_tol=1e-9)
```

**Failing input**: `x = [False, False, False]`, `y = [False, False, False]`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial import distance

x = np.array([False, False, False])
y = np.array([False, False, False])

result = distance.dice(x, y)

print(f"dice({x}, {y}) = {result}")
print(f"Expected: 0.0 (identical arrays should have distance 0)")
print(f"Actual: {result}")
```

Output:
```
dice([False False False], [False False False]) = nan
Expected: 0.0 (identical arrays should have distance 0)
Actual: nan
```

## Why This Is A Bug

1. **Mathematical expectation**: A distance metric between two identical arrays should be 0. The dice dissimilarity for identical inputs should be 0.0, not NaN.

2. **Inconsistency with similar metrics**: Other similar boolean distance metrics in scipy.spatial.distance handle this edge case correctly:
   - `jaccard([False, False], [False, False])` returns 0.0
   - `hamming([False, False], [False, False])` returns 0.0
   - `rogerstanimoto([False, False], [False, False])` returns 0.0

3. **Historical precedent**: The `jaccard` function had this exact same bug and it was fixed in SciPy v1.2.0, as documented in the jaccard docstring: "Previously, if all (positively weighted) elements in `u` and `v` are zero, the function would return `nan`. This was changed to return `0` instead."

4. **Real-world impact**: Users working with sparse boolean data (e.g., feature presence/absence in biology, document term vectors) would encounter NaN values when comparing empty feature sets, breaking downstream computations.

## Fix

The bug occurs in the final line of the `dice` function at line 1503 in `scipy/spatial/distance.py`:

```python
return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
```

When both arrays are all False, we have `ntt = ntf = nft = 0`, causing division by zero.

The fix should follow the same pattern as `jaccard`:

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1500,7 +1500,8 @@ def dice(u, v, w=None):
             ntt = (u * v * w).sum()
     (nft, ntf) = _nbool_correspond_ft_tf(u, v, w=w)
-    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
+    denominator = 2.0 * ntt + ntf + nft
+    return float((ntf + nft) / np.array(denominator)) if denominator != 0 else 0.0
```

This ensures that when both arrays are identical and all False (denominator = 0, numerator = 0), the function returns 0.0 instead of NaN.