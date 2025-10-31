# Bug Report: scipy.sparse.csgraph.laplacian LinearOperator matvec Returns Wrong Shape

**Target**: `scipy.sparse.csgraph.laplacian`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `scipy.sparse.csgraph.laplacian` is called with `form='lo'` to return a LinearOperator, the resulting LinearOperator crashes when performing matrix-vector multiplication due to incorrect output shape.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgraph


@given(
    st.integers(min_value=2, max_value=10),
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    st.integers(min_value=0, max_value=2**31 - 1),
    st.booleans()
)
@settings(max_examples=200)
def test_laplacian_form_consistency(n, max_weight, seed, normed):
    np.random.seed(seed)

    dense = np.random.rand(n, n) * max_weight
    dense = (dense + dense.T) / 2
    np.fill_diagonal(dense, 0)

    graph = csr_matrix(dense)

    lap_array = csgraph.laplacian(graph, normed=normed, form='array')
    lap_lo = csgraph.laplacian(graph, normed=normed, form='lo')

    test_vec = np.random.rand(n)

    result_array = lap_array @ test_vec
    result_lo = lap_lo @ test_vec

    assert np.allclose(result_array, result_lo, rtol=1e-9, atol=1e-9)
```

**Failing input**: `n=2, max_weight=1.0, seed=0, normed=False`

## Reproducing the Bug

```python
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgraph

graph = csr_matrix([[0, 1], [1, 0]], dtype=float)
lap_lo = csgraph.laplacian(graph, form='lo')
test_vec = np.array([1.0, 2.0])
result = lap_lo @ test_vec
```

Output:
```
ValueError: cannot reshape array of size 4 into shape (2,)
```

## Why This Is A Bug

The LinearOperator's `matvec` method is expected to take a 1D input vector of shape `(n,)` and return a 1D output vector of shape `(n,)`. However, the LinearOperator returned by `laplacian(..., form='lo')` returns a 2D array of shape `(n, n)` instead, causing scipy's LinearOperator wrapper to fail when trying to reshape the output.

The root cause is in the `_laplace` family of functions in `scipy/sparse/csgraph/_laplacian.py`. These functions use broadcasting operations like `v * d[:, np.newaxis]` which add an extra dimension. When `v` is a 1D array (as in matvec), this produces a 2D result, but the code doesn't handle this case differently from matmat where `v` is already 2D.

## Fix

```diff
--- a/scipy/sparse/csgraph/_laplacian.py
+++ b/scipy/sparse/csgraph/_laplacian.py
@@ -50,7 +50,10 @@ def _laplace(m, d):
-    return lambda v: v * d[:, np.newaxis] - m @ v
+    def matvec(v):
+        result = v * d[:, np.newaxis] - m @ v
+        return result.ravel() if v.ndim == 1 else result
+    return matvec


 def _laplace_normed(m, d, nd):
     laplace = _laplace(m, d)
-    return lambda v: nd[:, np.newaxis] * laplace(v * nd[:, np.newaxis])
+    def matvec(v):
+        result = nd[:, np.newaxis] * laplace(v * nd[:, np.newaxis])
+        return result.ravel() if v.ndim == 1 else result
+    return matvec


 def _laplace_normed_sym(m, d, nd):
     laplace_sym = _laplace_sym(m, d)
-    return lambda v: nd[:, np.newaxis] * laplace_sym(v * nd[:, np.newaxis])
+    def matvec(v):
+        result = nd[:, np.newaxis] * laplace_sym(v * nd[:, np.newaxis])
+        return result.ravel() if v.ndim == 1 else result
+    return matvec


 def _laplace_sym(m, d):
-    return (
-        lambda v: v * d[:, np.newaxis]
-        - m @ v
-        - np.transpose(np.conjugate(np.transpose(np.conjugate(v)) @ m))
-    )
+    def matvec(v):
+        result = (v * d[:, np.newaxis]
+                  - m @ v
+                  - np.transpose(np.conjugate(np.transpose(np.conjugate(v)) @ m)))
+        return result.ravel() if v.ndim == 1 else result
+    return matvec
```

Alternatively, a more targeted fix would be to modify `_linearoperator` to wrap the matvec function:

```diff
--- a/scipy/sparse/csgraph/_laplacian.py
+++ b/scipy/sparse/csgraph/_laplacian.py
@@ -38,7 +38,12 @@ def _linearoperator(mv, shape, dtype):
-    return LinearOperator(matvec=mv, matmat=mv, shape=shape, dtype=dtype)
+    def matvec_wrapper(v):
+        result = mv(v)
+        return result.ravel() if v.ndim == 1 else result
+
+    return LinearOperator(matvec=matvec_wrapper, matmat=mv, shape=shape, dtype=dtype)
```