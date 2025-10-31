# Bug Report: scipy.cluster.vq.kmeans2 Crashes on Square Data

**Target**: `scipy.cluster.vq.kmeans2`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`kmeans2` with `minit='random'` crashes with `LinAlgError: Matrix is not positive definite` when the number of observations equals the number of features (i.e., square data matrices).

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from scipy.cluster.vq import kmeans2
import numpy as np

@given(st.integers(min_value=5, max_value=30),
       st.integers(min_value=2, max_value=5),
       st.integers(min_value=2, max_value=8))
def test_kmeans2_different_minit_methods(n_obs, n_features, k):
    assume(k <= n_obs)
    data = np.random.randn(n_obs, n_features)
    centroids, labels = kmeans2(data, k, minit='random')
    assert centroids.shape == (k, n_features)
    assert len(labels) == n_obs
```

**Failing input**: `n_obs=5, n_features=5, k=2`

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import kmeans2

np.random.seed(0)
data = np.random.randn(5, 5)

centroids, labels = kmeans2(data, 2, minit='random')
```

**Output**:
```
numpy.linalg.LinAlgError: Matrix is not positive definite
```

## Why This Is A Bug

The `kmeans2` function should handle all valid input data shapes. When `n_observations == n_features`, the data is still valid for clustering, but the `minit='random'` initialization method crashes because it attempts to perform Cholesky decomposition on a covariance matrix that is often singular (not positive definite).

The root cause is in the `_krandinit` function (scipy/cluster/vq.py), which has special handling for rank-deficient covariance matrices when `data.shape[1] > data.shape[0]`, but fails to handle the case when `data.shape[1] == data.shape[0]`.

## Fix

```diff
--- a/scipy/cluster/vq.py
+++ b/scipy/cluster/vq.py
@@ -570,7 +570,7 @@ def _krandinit(data, k, rng, xp):
         x = xp.asarray(x)
         x *= xp.sqrt(_cov)
-    elif data.shape[1] > data.shape[0]:
+    elif data.shape[1] >= data.shape[0]:
         # initialize when the covariance matrix is rank deficient
         _, s, vh = xp.linalg.svd(data - mu, full_matrices=False)
         x = rng.standard_normal(size=(k, xp_size(s)))
```

This fix ensures that when the number of features is greater than or equal to the number of observations, the more robust SVD-based initialization is used instead of attempting Cholesky decomposition on a potentially singular covariance matrix.