# Bug Report: scipy.cluster.vq.kmeans2 Crashes with LinAlgError on Low-Variance Data

**Target**: `scipy.cluster.vq.kmeans2`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `kmeans2` function crashes with `numpy.linalg.LinAlgError: Matrix is not positive definite` when processing low-variance data using the default 'random' initialization method, even though the input data is valid according to the function's documented specifications.

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

if __name__ == "__main__":
    test_kmeans2_handles_low_variance_data()
```

<details>

<summary>
**Failing input**: `array with 9 identical rows and 1 slightly different row`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 32, in <module>
    test_kmeans2_handles_low_variance_data()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 9, in test_kmeans2_handles_low_variance_data
    npst.arrays(
               ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 22, in test_kmeans2_handles_low_variance_data
    centroids, labels = kmeans2(obs, k, iter=5, missing='raise')
                        ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_util.py", line 352, in wrapper
    return fun(*args, **kwargs)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/cluster/vq.py", line 817, in kmeans2
    code_book = init_meth(data, code_book, rng, xp)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/cluster/vq.py", line 571, in _krandinit
    x = x @ xp.linalg.cholesky(_cov).T
            ~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/array_api_compat/_internal.py", line 34, in wrapped_f
    return f(*args, xp=xp, **kwargs)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/array_api_compat/common/_linalg.py", line 89, in cholesky
    L = xp.linalg.cholesky(x, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/linalg/_linalg.py", line 897, in cholesky
    r = gufunc(a, signature=signature)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/linalg/_linalg.py", line 166, in _raise_linalgerror_nonposdef
    raise LinAlgError("Matrix is not positive definite")
numpy.linalg.LinAlgError: Matrix is not positive definite
Falsifying example: test_kmeans2_handles_low_variance_data(
    obs=array([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]),
    k=2,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/.local/lib/python3.13/site-packages/scipy/_lib/array_api_extra/_lib/_funcs.py:353
        /home/npc/.local/lib/python3.13/site-packages/scipy/_lib/array_api_extra/_lib/_funcs.py:361
        /home/npc/.local/lib/python3.13/site-packages/scipy/_lib/array_api_extra/_lib/_funcs.py:362
        /home/npc/.local/lib/python3.13/site-packages/scipy/cluster/vq.py:565
        /home/npc/.local/lib/python3.13/site-packages/scipy/cluster/vq.py:570
        (and 2 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import kmeans2
import sys
import traceback

# Create the test data - 10x10 array where 9 rows are identical (all 1s)
# and only the first row differs slightly (first element is 0)
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

print("Test data shape:", obs.shape)
print("Number of unique rows:", len(np.unique(obs, axis=0)))
print("Standard deviation:", np.std(obs))
print()

# Attempt to cluster with default 'random' initialization
print("Attempting kmeans2 with default 'random' initialization...")
print()

centroids, labels = kmeans2(obs, 2)
```

<details>

<summary>
LinAlgError: Matrix is not positive definite
</summary>
```
Test data shape: (10, 10)
Number of unique rows: 2
Standard deviation: 0.09949874371066199

Attempting kmeans2 with default 'random' initialization...

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/repo.py", line 28, in <module>
    centroids, labels = kmeans2(obs, 2)
                        ~~~~~~~^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_util.py", line 352, in wrapper
    return fun(*args, **kwargs)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/cluster/vq.py", line 817, in kmeans2
    code_book = init_meth(data, code_book, rng, xp)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/cluster/vq.py", line 571, in _krandinit
    x = x @ xp.linalg.cholesky(_cov).T
            ~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/array_api_compat/_internal.py", line 34, in wrapped_f
    return f(*args, xp=xp, **kwargs)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/array_api_compat/common/_linalg.py", line 89, in cholesky
    L = xp.linalg.cholesky(x, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/linalg/_linalg.py", line 897, in cholesky
    r = gufunc(a, signature=signature)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/linalg/_linalg.py", line 166, in _raise_linalgerror_nonposdef
    raise LinAlgError("Matrix is not positive definite")
numpy.linalg.LinAlgError: Matrix is not positive definite
```
</details>

## Why This Is A Bug

This violates expected behavior for multiple reasons:

1. **Valid Input Causes Crash**: The input array meets all documented requirements - it's a valid M×N ndarray with finite numeric values, proper shape, and sufficient observations (10 rows > k=2). The documentation states that `data` should be "A 'M' by 'N' array of 'M' observations in 'N' dimensions" with no restrictions on variance or data diversity.

2. **Undocumented Exception**: The function documentation only mentions `ClusterError` as a possible exception (when `missing='raise'` and empty clusters occur). There is no mention of `LinAlgError` being a possible exception, nor any warning about requirements for positive definite covariance matrices.

3. **Inconsistent Behavior Across Initialization Methods**: The same data works correctly with alternative initialization methods ('points' and '++'), but crashes with the default 'random' method. This inconsistency is not documented and violates the principle of least surprise.

4. **Poor Error Message**: The error "Matrix is not positive definite" originates from deep within the linear algebra stack and provides no actionable information to users about what went wrong or how to fix it.

5. **Reasonable Use Case**: Low-variance data legitimately occurs in real-world scenarios such as sensor measurements with stuck readings, controlled experiments with minimal variation, or datasets with many duplicate samples.

## Relevant Context

The crash occurs in the `_krandinit` function at line 571 of `/scipy/cluster/vq.py` during Cholesky decomposition of the covariance matrix. The 'random' initialization method attempts to generate k centroids from a Gaussian distribution with mean and variance estimated from the data. When the data has extremely low variance (like our test case with 9/10 identical rows), the covariance matrix becomes singular or near-singular, causing the Cholesky decomposition to fail.

The documentation describes the 'random' initialization as: "generate k centroids from a Gaussian with mean and variance estimated from the data" but does not warn about any mathematical prerequisites or failure modes for this method.

Relevant code location:
- File: `/scipy/cluster/vq.py`
- Function: `_krandinit` (lines 544-574)
- Failing line: 571 (`x = x @ xp.linalg.cholesky(_cov).T`)

Alternative initialization methods ('points' and '++') handle the same data without issues, demonstrating that the limitation is specific to the 'random' initialization implementation rather than a fundamental constraint of k-means clustering.

## Proposed Fix

```diff
--- a/scipy/cluster/vq.py
+++ b/scipy/cluster/vq.py
@@ -563,13 +563,24 @@ def _krandinit(data, k, rng, xp):
         x = xp.asarray(x)
         x = x @ sVh
     else:
-        _cov = xpx.atleast_nd(xpx.cov(data.T, xp=xp), ndim=2, xp=xp)
-
-        # k rows, d cols (one row = one obs)
-        # Generate k sample of a random variable ~ Gaussian(mu, cov)
-        x = rng.standard_normal(size=(k, xp_size(mu)))
-        x = xp.asarray(x)
-        x = x @ xp.linalg.cholesky(_cov).T
+        try:
+            _cov = xpx.atleast_nd(xpx.cov(data.T, xp=xp), ndim=2, xp=xp)
+
+            # Check if covariance matrix is well-conditioned
+            cond_number = xp.linalg.cond(_cov)
+            if cond_number > 1e10:
+                # Covariance matrix is ill-conditioned, fall back to simpler initialization
+                return _kpoints(data, k, rng, xp)
+
+            # k rows, d cols (one row = one obs)
+            # Generate k sample of a random variable ~ Gaussian(mu, cov)
+            x = rng.standard_normal(size=(k, xp_size(mu)))
+            x = xp.asarray(x)
+            x = x @ xp.linalg.cholesky(_cov).T
+        except (xp.linalg.LinAlgError, np.linalg.LinAlgError):
+            # Cholesky decomposition failed due to singular matrix
+            # Fall back to 'points' initialization method
+            return _kpoints(data, k, rng, xp)

     x += mu
     return x
```