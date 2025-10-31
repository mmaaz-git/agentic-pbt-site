# Bug Report: scipy.cluster.vq.kmeans2 Crash on Small Datasets

**Target**: `scipy.cluster.vq.kmeans2`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`kmeans2` crashes with `numpy.linalg.LinAlgError: Matrix is not positive definite` when called with small datasets using the default `minit='random'` initialization method.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.cluster.vq import kmeans2


@given(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=5))
@settings(max_examples=500)
def test_kmeans2_should_not_crash(n_obs, n_features, k):
    if k > n_obs:
        return

    obs = np.random.randn(n_obs, n_features)
    centroid, label = kmeans2(obs, k)
```

**Failing input**: `n_obs=2, n_features=2, k=2`

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import kmeans2

np.random.seed(0)
obs = np.random.randn(2, 2)

centroid, label = kmeans2(obs, 2)
```

Output:
```
numpy.linalg.LinAlgError: Matrix is not positive definite
```

Workaround using different initialization:
```python
centroid, label = kmeans2(obs, 2, minit='points')
```

## Why This Is A Bug

The function accepts valid input (2 observations with 2 features, clustering into 2 clusters) but crashes instead of either:
1. Handling the case gracefully
2. Providing a clear error message
3. Automatically falling back to a different initialization method

The default `minit='random'` initialization computes a Cholesky decomposition of the covariance matrix in `_krandinit`, which fails when the matrix is not positive definite. This commonly occurs with small datasets where n_observations ≈ n_features.

The crash violates the principle of least surprise - users expect the function to either work or provide a helpful error message, not crash with an obscure linear algebra error.

## Fix

The fix should catch the `LinAlgError` in the initialization and either fall back to a more robust initialization method or provide a clear error message. Here's a suggested patch:

```diff
--- a/scipy/cluster/vq.py
+++ b/scipy/cluster/vq.py
@@ -568,7 +568,14 @@ def _krandinit(data, k, rng, xp):
     mu = xp.mean(data, axis=0)
     _cov = xp.cov(data.T)
     x = rng.standard_normal((k, mu.shape[0]))
-    x = x @ xp.linalg.cholesky(_cov).T
+    try:
+        x = x @ xp.linalg.cholesky(_cov).T
+    except (xp.linalg.LinAlgError, np.linalg.LinAlgError):
+        # Covariance matrix is not positive definite (common with small datasets)
+        # Fall back to sampling from per-feature standard deviation
+        std = xp.std(data, axis=0)
+        std = xp.where(std > 0, std, 1.0)
+        x = x * std
     x = x + mu
     return x
```