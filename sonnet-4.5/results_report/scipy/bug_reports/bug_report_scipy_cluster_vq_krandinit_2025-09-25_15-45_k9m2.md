# Bug Report: scipy.cluster.vq Random Initialization Cholesky Failure

**Target**: `scipy.cluster.vq._krandinit`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The random initialization method (`_krandinit`) for k-means crashes with `LinAlgError: Matrix is not positive definite` when the input data contains duplicate rows or is rank-deficient, because it attempts Cholesky decomposition on a singular covariance matrix.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.cluster.vq import kmeans2


@given(st.integers(min_value=2, max_value=20),
       st.integers(min_value=2, max_value=10))
@settings(max_examples=100)
def test_kmeans2_random_init_handles_rank_deficient_data(n_obs, n_features):
    data = np.random.randn(n_obs, 1)
    data = np.tile(data, (1, n_features))

    centroid, label = kmeans2(data, min(2, n_obs), minit='random')

    assert centroid.shape[1] == n_features
```

**Failing input**: Any data with duplicate rows or rank-deficient covariance, e.g., `data = [[1, 2], [1, 2], [1, 2], [1, 2]]`

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import kmeans2

data = np.array([
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0],
])

try:
    centroid, label = kmeans2(data, 2, minit='random')
except np.linalg.LinAlgError as e:
    print(f"Crash: {e}")
```

Output:
```
Crash: Matrix is not positive definite
```

## Why This Is A Bug

The `_krandinit` function initializes centroids by sampling from a Gaussian distribution with mean and covariance estimated from the data. At line 571 in `scipy/cluster/vq.py`:

```python
_cov = xpx.atleast_nd(xpx.cov(data.T, xp=xp), ndim=2, xp=xp)
x = rng.standard_normal(size=(k, xp_size(mu)))
x = xp.asarray(x)
x = x @ xp.linalg.cholesky(_cov).T  # Crashes here
```

When the data has duplicate rows or is otherwise rank-deficient, the covariance matrix `_cov` is singular (not positive definite), causing Cholesky decomposition to fail.

The code already handles the rank-deficient case when `data.shape[1] > data.shape[0]` (lines 557-563) using SVD, but fails to handle it for the common case when `data.shape[1] <= data.shape[0]`.

This is a severe issue because:
1. Duplicate data points are common in real-world datasets
2. The crash prevents users from running k-means on valid input
3. Other initialization methods ('points', '++') work fine on the same data

## Fix

Use SVD-based initialization for all cases, not just when `data.shape[1] > data.shape[0]`, or add a try-except to fall back to SVD when Cholesky fails:

```diff
--- a/scipy/cluster/vq.py
+++ b/scipy/cluster/vq.py
@@ -563,12 +563,19 @@ def _krandinit(data, k, rng, xp):
         x = x @ sVh
     else:
         _cov = xpx.atleast_nd(xpx.cov(data.T, xp=xp), ndim=2, xp=xp)

         # k rows, d cols (one row = one obs)
         # Generate k sample of a random variable ~ Gaussian(mu, cov)
         x = rng.standard_normal(size=(k, xp_size(mu)))
         x = xp.asarray(x)
-        x = x @ xp.linalg.cholesky(_cov).T
+        try:
+            x = x @ xp.linalg.cholesky(_cov).T
+        except np.linalg.LinAlgError:
+            # Covariance matrix is not positive definite (e.g., duplicate data)
+            # Fall back to SVD-based approach
+            _, s, vh = xp.linalg.svd(data - mu, full_matrices=False)
+            sVh = s[:, None] * vh / xp.sqrt(data.shape[0] - xp.asarray(1.))
+            x = x[:, :len(s)] @ sVh

     x += mu
     return x
```