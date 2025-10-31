# Bug Report: scipy.cluster.vq.kmeans2 Crashes with LinAlgError on Square Data Matrices

**Target**: `scipy.cluster.vq.kmeans2`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `kmeans2` function with `minit='random'` crashes with a `LinAlgError: Matrix is not positive definite` when the input data has equal numbers of observations and features (square matrices like 5x5).

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

if __name__ == "__main__":
    test_kmeans2_different_minit_methods()
```

<details>

<summary>
**Failing input**: `n_obs=5, n_features=5, k=2`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/_lib/_util.py:352: UserWarning: One of the clusters is empty. Re-run kmeans with a different initialization.
  return fun(*args, **kwargs)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 16, in <module>
    test_kmeans2_different_minit_methods()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 6, in test_kmeans2_different_minit_methods
    st.integers(min_value=2, max_value=5),
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 11, in test_kmeans2_different_minit_methods
    centroids, labels = kmeans2(data, k, minit='random')
                        ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
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
Falsifying example: test_kmeans2_different_minit_methods(
    n_obs=5,
    n_features=5,
    k=2,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/linalg/_linalg.py:166
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import kmeans2

np.random.seed(0)
data = np.random.randn(5, 5)

centroids, labels = kmeans2(data, 2, minit='random')
```

<details>

<summary>
LinAlgError: Matrix is not positive definite
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/repo.py", line 7, in <module>
    centroids, labels = kmeans2(data, 2, minit='random')
                        ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
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

The `kmeans2` function should handle all valid input data shapes according to its documentation, which states it accepts "M by N array of M observations in N dimensions" with no restrictions on the relationship between M and N. When the number of observations equals the number of features (M == N), the resulting covariance matrix is rank-deficient (rank at most M-1), making it not positive definite. The `minit='random'` initialization incorrectly attempts Cholesky decomposition on this singular matrix, causing a crash.

The code in `scipy/cluster/vq.py` already has special handling for rank-deficient cases when `data.shape[1] > data.shape[0]` (line 557), using SVD-based initialization instead of Cholesky decomposition. However, it fails to handle the boundary case when `data.shape[1] == data.shape[0]`, falling through to the else block that assumes the covariance matrix is full rank. This is an off-by-one logic error in the conditional check.

Other initialization methods (`minit='points'` and `minit='++'`) work correctly with square matrices, making this inconsistent behavior that violates the principle of least surprise.

## Relevant Context

- The bug occurs in the `_krandinit` function at line 571 of `/scipy/cluster/vq.py`
- Square data matrices are common in practice (correlation matrices, distance matrices, feature spaces where dimensions match sample size)
- The mathematical issue: For an n×n data matrix, the sample covariance matrix has rank at most n-1, making it singular
- The existing SVD-based approach (lines 559-563) already handles rank-deficient matrices correctly
- Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html
- Source code: https://github.com/scipy/scipy/blob/main/scipy/cluster/vq.py

## Proposed Fix

```diff
--- a/scipy/cluster/vq.py
+++ b/scipy/cluster/vq.py
@@ -554,7 +554,7 @@ def _krandinit(data, k, rng, xp):
         x = rng.standard_normal(size=k)
         x = xp.asarray(x)
         x *= xp.sqrt(_cov)
-    elif data.shape[1] > data.shape[0]:
+    elif data.shape[1] >= data.shape[0]:
         # initialize when the covariance matrix is rank deficient
         _, s, vh = xp.linalg.svd(data - mu, full_matrices=False)
         x = rng.standard_normal(size=(k, xp_size(s)))
```