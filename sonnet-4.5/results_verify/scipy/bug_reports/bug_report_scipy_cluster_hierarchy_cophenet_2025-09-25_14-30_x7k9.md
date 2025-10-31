# Bug Report: scipy.cluster.hierarchy.cophenet Returns NaN Without Error or Documentation

**Target**: `scipy.cluster.hierarchy.cophenet`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `cophenet` function returns `nan` for valid inputs (n=2 observations or constant distance data) without raising an error or documenting this behavior, violating the function's implicit contract with users.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
import numpy as np

@given(st.integers(min_value=2, max_value=15),
       st.integers(min_value=1, max_value=10))
def test_cophenet_correlation_in_range(n_obs, n_features):
    data = np.random.randn(n_obs, n_features)
    y = pdist(data)
    Z = linkage(y, method='single')
    c, coph_dists = cophenet(Z, y)

    assert -1.0 <= c <= 1.0, f"Cophenetic correlation {c} should be in [-1, 1]"
```

**Failing input**: `n_obs=2, n_features=1` (or any value with constant distances)

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

data = np.array([[0.0], [1.0]])
y = pdist(data)
Z = linkage(y, method='single')
c, coph_dists = cophenet(Z, y)
print(f"Cophenetic correlation: {c}")

data_const = np.ones((5, 2))
y_const = pdist(data_const)
Z_const = linkage(y_const, method='single')
c_const, coph_const = cophenet(Z_const, y_const)
print(f"Constant data correlation: {c_const}")
```

## Why This Is A Bug

The function silently returns `nan` in two cases:
1. When n=2 (only 1 distance pair, cannot compute correlation)
2. When all distances are constant (zero variance, correlation mathematically undefined)

The function contract is violated because:
- The documentation does not mention that `nan` can be returned
- No error is raised for these edge cases
- Only a RuntimeWarning is emitted (`invalid value encountered in scalar divide`)
- Users cannot distinguish between these edge cases and actual errors

The correlation is a bounded value in [-1, 1], so returning `nan` without documentation is surprising and can cause downstream errors in user code that doesn't check for `nan`.

## Fix

The function should handle these edge cases explicitly. Here's a suggested fix:

```diff
--- a/scipy/cluster/hierarchy.py
+++ b/scipy/cluster/hierarchy.py
@@ -1710,8 +1710,17 @@ def cophenet(Z, Y=None):
     denomA = (dist - xp.mean(dist)) ** 2
     denomB = (coph_dists - xp.mean(coph_dists)) ** 2
     numerator = (dist - xp.mean(dist)) * (coph_dists - xp.mean(coph_dists))
+
+    denom_product = xp.sum(denomA) * xp.sum(denomB)
+
+    if denom_product == 0:
+        if xp.sum(denomA) == 0 and xp.sum(denomB) == 0:
+            return 1.0, coph_dists
+        else:
+            return 0.0, coph_dists
+
     c = xp.sum(numerator) / xp.sqrt(xp.sum(denomA) * xp.sum(denomB))
+
     return c, coph_dists
```

Alternatively, the function could raise a `ValueError` with a clear message:

```python
if denom_product == 0 or len(dist) < 2:
    raise ValueError(
        "Cannot compute cophenetic correlation: "
        "requires at least 2 distance pairs with non-zero variance"
    )
```

Or at minimum, the documentation should be updated to mention this behavior:

```diff
+    Returns
+    -------
+    c : float
+        The cophenetic correlation coefficient. Returns ``nan`` if the
+        distance matrix has fewer than 2 pairs or zero variance.
```