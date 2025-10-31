# Bug Report: scipy.cluster.vq.kmeans2 Crashes on Rank-Deficient Data with Random Initialization

**Target**: `scipy.cluster.vq.kmeans2`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `kmeans2` function crashes with `LinAlgError: Matrix is not positive definite` when using `minit='random'` on rank-deficient data (data with linearly dependent features), even though such data is valid input according to the documentation.

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

if __name__ == "__main__":
    test_kmeans2_cholesky_crash_on_degenerate_data()
```

<details>

<summary>
**Failing input**: `obs=array([[0.0, 0.0, 0.0], ...]), k=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 28, in test_kmeans2_cholesky_crash_on_degenerate_data
    codebook, labels = kmeans2(obs_degenerate, k, iter=1, minit='random')
                       ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 36, in <module>
    test_kmeans2_cholesky_crash_on_degenerate_data()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 19, in test_kmeans2_cholesky_crash_on_degenerate_data
    k=st.integers(min_value=1, max_value=10))
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 32, in test_kmeans2_cholesky_crash_on_degenerate_data
    pytest.fail(f"kmeans2 with minit='random' crashes on rank-deficient data: {e}")
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: kmeans2 with minit='random' crashes on rank-deficient data: Matrix is not positive definite
Falsifying example: test_kmeans2_cholesky_crash_on_degenerate_data(
    # The test always failed when commented parts were varied together.
    obs=array(
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]],
    ),  # or any other generated value
    k=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import kmeans2

# Create rank-deficient data where all features are perfectly correlated
data = np.array([[1.0, 1.0, 1.0],
                 [2.0, 2.0, 2.0],
                 [3.0, 3.0, 3.0],
                 [4.0, 4.0, 4.0],
                 [5.0, 5.0, 5.0]])

print("Testing scipy.cluster.vq.kmeans2 with rank-deficient data")
print("Data shape:", data.shape)
print("Data:\n", data)
print("\nAttempting kmeans2 with k=2, minit='random'...")

try:
    codebook, labels = kmeans2(data, 2, minit='random')
    print("Success! Codebook:", codebook)
    print("Labels:", labels)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
```

<details>

<summary>
Error occurred: LinAlgError: Matrix is not positive definite
</summary>
```
Testing scipy.cluster.vq.kmeans2 with rank-deficient data
Data shape: (5, 3)
Data:
 [[1. 1. 1.]
 [2. 2. 2.]
 [3. 3. 3.]
 [4. 4. 4.]
 [5. 5. 5.]]

Attempting kmeans2 with k=2, minit='random'...
Error occurred: LinAlgError: Matrix is not positive definite
```
</details>

## Why This Is A Bug

This crash violates expected behavior for several reasons:

1. **Documentation doesn't restrict input**: The kmeans2 documentation states it accepts "an 'M' by 'N' array of 'M' observations in 'N' dimensions" without requiring features to be uncorrelated or the covariance matrix to be positive-definite.

2. **Inconsistent behavior across initialization methods**: The same rank-deficient data works correctly with `minit='points'` and `minit='++'`, but crashes with `minit='random'`. Users reasonably expect all initialization methods to accept the same valid input.

3. **Valid real-world scenario**: Rank-deficient data commonly occurs in practice:
   - Duplicated or highly correlated features in datasets
   - Scaled versions of the same measurement
   - One-hot encoded categorical variables with redundancy
   - Constant features across some dimensions

4. **Implementation oversight**: The crash occurs in `_krandinit` (scipy/cluster/vq.py:571) when computing Cholesky decomposition of the covariance matrix. The code already handles rank-deficient covariance when `data.shape[1] > data.shape[0]` using SVD (lines 557-563), but incorrectly assumes rank-deficiency only occurs in that case. Rank-deficiency can occur whenever features are linearly dependent, regardless of dimensions.

5. **Cholesky requirements not met**: Cholesky decomposition requires a positive-definite matrix, but rank-deficient data produces a singular covariance matrix that is only positive-semidefinite. The function should handle this gracefully rather than crashing.

## Relevant Context

The `_krandinit` function's documentation states it generates centroids "from a Gaussian with mean and variance estimated from the data." This should be possible even for rank-deficient data using techniques like SVD or pseudoinverse, as the code already does in the `data.shape[1] > data.shape[0]` case.

Key code location: `/scipy/cluster/vq.py`, lines 540-574 (`_krandinit` function)

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html

The issue affects users trying to cluster data with correlated features, a common scenario in data analysis, especially when feature engineering hasn't removed redundant dimensions.

## Proposed Fix

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
+            # k rows, d cols (one row = one obs)
+            # Generate k sample of a random variable ~ Gaussian(mu, cov)
+            x = rng.standard_normal(size=(k, xp_size(mu)))
+            x = xp.asarray(x)
+            x = x @ xp.linalg.cholesky(_cov).T
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