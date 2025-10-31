# Bug Report: scipy.cluster.vq.kmeans2 Crashes on Low-Variance Data

**Target**: `scipy.cluster.vq.kmeans2`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `kmeans2` function crashes with `LinAlgError: Matrix is not positive definite` when clustering data with low variance using the default 'random' initialization method.

## Property-Based Test

```python
import numpy as np
from scipy.cluster.vq import kmeans2
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst


@given(
    npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=50),
        elements=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)
    ),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_kmeans2_handles_low_variance_data(obs, k):
    assume(obs.shape[0] > k)
    assume(np.std(obs) > 1e-6)

    try:
        centroids, labels = kmeans2(obs, k, iter=5, missing='raise')
        assert centroids.shape[0] <= k
        assert centroids.shape[1] == obs.shape[1]
    except Exception as e:
        if "empty" in str(e).lower():
            pass
        else:
            raise
```

**Failing input**: Nearly constant data with 9/10 rows identical

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import kmeans2

obs = np.array([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

kmeans2(obs, 2)
```

Output:
```
numpy.linalg.LinAlgError: Matrix is not positive definite
```

## Why This Is A Bug

The `kmeans2` function violates the principle of graceful degradation when encountering valid input data:

1. **Valid input**: The data is a valid M×N observation array as specified in the documentation
2. **No documented precondition**: The function does not document that it requires non-singular covariance matrices
3. **Reasonable use case**: Users may legitimately attempt to cluster data with low variance (e.g., sensor data with stuck readings, controlled experiments, nearly identical samples)
4. **Inconsistent behavior**: Other initialization methods ('points', '++') handle this data gracefully, but the default 'random' method crashes
5. **Poor error message**: The "Matrix is not positive definite" error from deep in the call stack doesn't help users understand what went wrong

## Fix

The issue occurs in the `_krandinit` function (vq.py:571) which attempts Cholesky decomposition on a near-singular covariance matrix. The fix should add proper error handling or use a more robust decomposition method:

```diff
--- a/scipy/cluster/vq.py
+++ b/scipy/cluster/vq.py
@@ -563,7 +563,15 @@ def _krandinit(data, k, rng, xp):
         x = xp.asarray(x)
         x = x @ sVh
     else:
-        _cov = xpx.atleast_nd(xpx.cov(data.T, xp=xp), ndim=2, xp=xp)
+        try:
+            _cov = xpx.atleast_nd(xpx.cov(data.T, xp=xp), ndim=2, xp=xp)
+            # Check if covariance matrix is well-conditioned
+            if xp.linalg.cond(_cov) > 1e10:
+                # Fall back to simpler initialization for ill-conditioned data
+                return _kpoints(data, k, rng, xp)
+        except np.linalg.LinAlgError:
+            # Covariance matrix is singular, use simpler initialization
+            return _kpoints(data, k, rng, xp)

         # k rows, d cols (one row = one obs)
         # Generate k sample of a random variable ~ Gaussian(mu, cov)
@@ -571,6 +579,9 @@ def _krandinit(data, k, rng, xp):
         x = xp.asarray(x)
         x = x @ xp.linalg.cholesky(_cov).T

+    except np.linalg.LinAlgError:
+        # Cholesky decomposition failed, use simpler initialization
+        return _kpoints(data, k, rng, xp)
+
     x += mu
     return x
```