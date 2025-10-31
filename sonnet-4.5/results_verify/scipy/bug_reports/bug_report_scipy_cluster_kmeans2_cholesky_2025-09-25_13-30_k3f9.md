# Bug Report: scipy.cluster.vq.kmeans2 Cholesky Crash

**Target**: `scipy.cluster.vq.kmeans2`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`kmeans2` crashes with `numpy.linalg.LinAlgError: Matrix is not positive definite` when using the default 'random' initialization method on valid input data. The crash occurs in the `_krandinit` function which attempts a Cholesky decomposition of the covariance matrix without checking if the matrix is positive definite.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from scipy.cluster.vq import kmeans2


@given(st.integers(min_value=3, max_value=20),
       st.integers(min_value=1, max_value=5),
       st.integers(min_value=2, max_value=5))
@settings(max_examples=100)
def test_kmeans2_returns_k_clusters(n_obs, n_features, k):
    assume(k <= n_obs)

    data = np.random.randn(n_obs, n_features)

    centroids, labels = kmeans2(data, k)

    assert len(centroids) == k
    assert len(labels) == n_obs
    assert np.all(labels >= 0) and np.all(labels < k)
```

**Failing input**: `n_obs=3, n_features=3, k=2` (and other small inputs)

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import kmeans2

np.random.seed(0)
data = np.random.randn(3, 3)
k = 2

centroids, labels = kmeans2(data, k, minit='random')
```

Output:
```
numpy.linalg.LinAlgError: Matrix is not positive definite
```

The crash occurs in `scipy/cluster/vq.py` in the `_krandinit` function at line:
```python
x = x @ xp.linalg.cholesky(_cov).T
```

## Why This Is A Bug

1. **Valid input rejected**: The input data is perfectly valid - a 3x3 array of random numbers is a legitimate input for kmeans2.

2. **Default behavior fails**: The crash happens with the default initialization method ('random'). Users should not experience crashes with default parameters on valid inputs.

3. **Inconsistent error handling**: The `_krandinit` function already handles rank-deficient cases when `data.shape[1] > data.shape[0]` using SVD, but fails to handle similar numerical issues in the general case.

4. **Poor user experience**: The error message "Matrix is not positive definite" is cryptic and doesn't suggest a solution. Users don't expect linear algebra errors from a clustering function.

## Fix

Replace the Cholesky decomposition with a more robust approach. The function already uses SVD for rank-deficient cases - the same approach can be used generally:

```diff
--- a/scipy/cluster/vq.py
+++ b/scipy/cluster/vq.py
@@ -566,11 +566,19 @@ def _krandinit(data, k, rng, xp):
         x = x @ sVh
     else:
         _cov = xpx.atleast_nd(xpx.cov(data.T, xp=xp), ndim=2, xp=xp)
-
-        # k rows, d cols (one row = one obs)
-        # Generate k sample of a random variable ~ Gaussian(mu, cov)
-        x = rng.standard_normal(size=(k, xp_size(mu)))
-        x = xp.asarray(x)
-        x = x @ xp.linalg.cholesky(_cov).T
+
+        try:
+            x = rng.standard_normal(size=(k, xp_size(mu)))
+            x = xp.asarray(x)
+            x = x @ xp.linalg.cholesky(_cov).T
+        except np.linalg.LinAlgError:
+            # Covariance matrix is not positive definite, use SVD instead
+            _, s, vh = xp.linalg.svd(_cov, full_matrices=False)
+            x = rng.standard_normal(size=(k, xp_size(mu)))
+            x = xp.asarray(x)
+            sVh = xp.sqrt(s[:, None]) * vh
+            x = x @ sVh

     x += mu
     return x
```

Alternatively, always use SVD instead of Cholesky for better numerical stability.