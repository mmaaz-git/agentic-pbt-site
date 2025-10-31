# Bug Report: scipy.cluster.vq.kmeans2 Crashes on Rank-Deficient Data

**Target**: `scipy.cluster.vq.kmeans2`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`kmeans2` with `minit='random'` crashes with `LinAlgError: Matrix is not positive definite` when given data where all features are perfectly correlated (rank-deficient data), even though this is valid input.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy.cluster.vq import kmeans2
import pytest


def fixed_size_arrays(min_rows, max_rows, num_cols, min_val=-1e6, max_val=1e6):
    return st.builds(
        np.array,
        st.lists(
            st.lists(st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False),
                    min_size=num_cols, max_size=num_cols),
            min_size=min_rows, max_size=max_rows
        )
    )


@given(obs=fixed_size_arrays(10, 50, 3, -100, 100),
       k=st.integers(min_value=1, max_value=10))
@settings(max_examples=100)
def test_kmeans2_cholesky_crash_on_degenerate_data(obs, k):
    assume(len(obs) >= k)

    obs_degenerate = np.zeros_like(obs)
    obs_degenerate[:, :] = obs[:, 0:1]

    try:
        codebook, labels = kmeans2(obs_degenerate, k, iter=1, minit='random')
        assert True
    except np.linalg.LinAlgError as e:
        if "not positive definite" in str(e):
            pytest.fail(f"kmeans2 with minit='random' crashes on rank-deficient data: {e}")
        raise
```

**Failing input**: Any data where all features are perfectly correlated, e.g., `[[1, 1, 1], [2, 2, 2], [3, 3, 3]]`

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import kmeans2

data = np.array([[1.0, 1.0, 1.0],
                 [2.0, 2.0, 2.0],
                 [3.0, 3.0, 3.0],
                 [4.0, 4.0, 4.0],
                 [5.0, 5.0, 5.0]])

codebook, labels = kmeans2(data, 2, minit='random')
```

Output:
```
numpy.linalg.LinAlgError: Matrix is not positive definite
```

## Why This Is A Bug

The crash occurs in `_krandinit` (vq.py:571) when computing Cholesky decomposition:
```python
x = x @ xp.linalg.cholesky(_cov).T
```

When all features are perfectly correlated, the covariance matrix is singular (rank-deficient), making it only positive-semidefinite, not positive-definite. Cholesky decomposition requires positive-definite matrices.

The code already handles rank-deficient covariance when `data.shape[1] > data.shape[0]` (lines 557-563) using SVD, but fails to handle rank-deficiency when `data.shape[1] <= data.shape[0]`. This is an incorrect assumption: rank-deficiency can occur regardless of the number of observations when features are correlated.

This is valid input because:
1. Nothing in the documentation requires features to be uncorrelated
2. Real-world data often has correlated or redundant features
3. Other initialization methods ('points', '++', 'matrix') handle this data correctly

## Fix

The fix should check for rank-deficiency before attempting Cholesky decomposition, and fall back to SVD-based initialization similar to lines 557-563:

```diff
--- a/scipy/cluster/vq.py
+++ b/scipy/cluster/vq.py
@@ -563,11 +563,20 @@ def _krandinit(data, k, rng, xp):
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
+        # Check if covariance matrix is positive definite
+        try:
+            L = xp.linalg.cholesky(_cov)
+            x = rng.standard_normal(size=(k, xp_size(mu)))
+            x = xp.asarray(x)
+            x = x @ L.T
+        except np.linalg.LinAlgError:
+            # Fall back to SVD for rank-deficient covariance
+            _, s, vh = xp.linalg.svd(data - mu, full_matrices=False)
+            x = rng.standard_normal(size=(k, xp_size(s)))
+            x = xp.asarray(x)
+            sVh = s[:, None] * vh / xp.sqrt(data.shape[0] - xp.asarray(1.))
+            x = x @ sVh

     x += mu
     return x
```