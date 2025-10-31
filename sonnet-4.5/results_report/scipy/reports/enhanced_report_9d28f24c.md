# Bug Report: scipy.sparse.csgraph.laplacian LinearOperator Crashes on Matrix-Vector Multiplication

**Target**: `scipy.sparse.csgraph.laplacian`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The LinearOperator returned by `scipy.sparse.csgraph.laplacian` with `form='lo'` crashes when performing matrix-vector multiplication with 1D vectors, violating the scipy LinearOperator interface contract.

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

# Run the test
if __name__ == "__main__":
    test_laplacian_form_consistency()
```

<details>

<summary>
**Failing input**: `n=2, max_weight=1.0, seed=0, normed=False`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 35, in <module>
    test_laplacian_form_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 8, in test_laplacian_form_consistency
    st.integers(min_value=2, max_value=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 29, in test_laplacian_form_consistency
    result_lo = lap_lo @ test_vec
                ~~~~~~~^~~~~~~~~~
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_interface.py", line 479, in __matmul__
    return self.__mul__(other)
           ~~~~~~~~~~~~^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_interface.py", line 436, in __mul__
    return self.dot(x)
           ~~~~~~~~^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_interface.py", line 469, in dot
    return self.matvec(x)
           ~~~~~~~~~~~^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_interface.py", line 266, in matvec
    y = y.reshape(M)
ValueError: cannot reshape array of size 4 into shape (2,)
Falsifying example: test_laplacian_form_consistency(
    n=2,
    max_weight=1.0,
    seed=0,
    normed=False,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgraph

# Create a simple 2x2 symmetric graph
graph = csr_matrix([[0, 1], [1, 0]], dtype=float)

# Create LinearOperator with form='lo'
lap_lo = csgraph.laplacian(graph, form='lo')

# Create test vector
test_vec = np.array([1.0, 2.0])

# Attempt matrix-vector multiplication - this should crash
try:
    result = lap_lo @ test_vec
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# Also test with matvec directly
print("\nTesting with matvec directly:")
try:
    result = lap_lo.matvec(test_vec)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# Compare with form='array' which should work
print("\nComparing with form='array':")
lap_array = csgraph.laplacian(graph, form='array')
result_array = lap_array @ test_vec
print(f"Result from array form: {result_array}")
print(f"Shape: {result_array.shape}")

# Test with 2D input (which should work)
print("\nTesting with 2D input:")
test_vec_2d = np.array([[1.0], [2.0]])
try:
    result_2d = lap_lo @ test_vec_2d
    print(f"Result from 2D input: {result_2d}")
    print(f"Shape: {result_2d.shape}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError: cannot reshape array of size 4 into shape (2,)
</summary>
```
Error occurred: ValueError: cannot reshape array of size 4 into shape (2,)

Testing with matvec directly:
Error occurred: ValueError: cannot reshape array of size 4 into shape (2,)

Comparing with form='array':
Result from array form: [-1.  1.]
Shape: (2,)

Testing with 2D input:
Result from 2D input: [[-1.]
 [ 1.]]
Shape: (2, 1)
```
</details>

## Why This Is A Bug

This violates the scipy.sparse.linalg.LinearOperator interface specification which explicitly requires that the `matvec` method must handle both (N,) and (N,1) input shapes correctly. According to the LinearOperator documentation, when given a 1D input vector of shape (N,), matvec should return a 1D output vector of shape (M,).

The bug occurs because the lambda functions in `_laplacian.py` lines 381-400 use broadcasting operations like `v * d[:, np.newaxis]` which add an extra dimension. When `v` is a 1D array (shape (n,)), this operation produces a 2D array (shape (n,n)) instead of the expected 1D array. The LinearOperator wrapper then fails when trying to reshape this 2D result back to 1D.

The same lambda function is incorrectly used for both `matvec` (expects 1D->1D) and `matmat` (expects 2D->2D) operations at line 404: `LinearOperator(matvec=mv, matmat=mv, shape=shape, dtype=dtype)`, causing the dimension mismatch for 1D inputs.

## Relevant Context

The bug affects all four Laplacian computation variants:
- `_laplace` (line 381-382): Basic Laplacian
- `_laplace_normed` (line 385-387): Normalized Laplacian
- `_laplace_sym` (line 390-395): Symmetric Laplacian
- `_laplace_normed_sym` (line 398-400): Normalized symmetric Laplacian

This is particularly problematic because:
1. LinearOperator is designed to be memory-efficient for large sparse matrices
2. It's commonly used with iterative solvers (LOBPCG, CG, GMRES) which rely on matvec operations
3. The documentation specifically demonstrates LinearOperator usage (line 256-262)
4. The same operation works correctly with `form='array'`, showing inconsistent behavior

Source code location: `/scipy/sparse/csgraph/_laplacian.py`
LinearOperator interface: `scipy.sparse.linalg._interface.py:266`

## Proposed Fix

```diff
--- a/scipy/sparse/csgraph/_laplacian.py
+++ b/scipy/sparse/csgraph/_laplacian.py
@@ -378,27 +378,43 @@ def _setdiag_dense(m, d):
     m.flat[::step] = d


 def _laplace(m, d):
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


 def _laplace_normed_sym(m, d, nd):
     laplace_sym = _laplace_sym(m, d)
-    return lambda v: nd[:, np.newaxis] * laplace_sym(v * nd[:, np.newaxis])
+    def matvec(v):
+        result = nd[:, np.newaxis] * laplace_sym(v * nd[:, np.newaxis])
+        return result.ravel() if v.ndim == 1 else result
+    return matvec


 def _linearoperator(mv, shape, dtype):
```