# Bug Report: scipy.sparse.linalg.inv Returns ndarray for 1x1 Matrices

**Target**: `scipy.sparse.linalg.inv`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `scipy.sparse.linalg.inv` function returns a `numpy.ndarray` instead of a sparse array when given a 1x1 sparse matrix, violating its documented contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

@given(st.integers(min_value=1, max_value=10), st.random_module())
@settings(max_examples=100)
def test_inv_double_inversion(n, random):
    A_dense = np.random.rand(n, n) + np.eye(n) * 2
    A = sp.csc_array(A_dense)
    assume(np.linalg.det(A_dense) > 0.01)

    Ainv = sla.inv(A)
    Ainvinv = sla.inv(Ainv)

    result = Ainvinv.toarray()
    expected = A.toarray()
    assert np.allclose(result, expected, rtol=1e-5, atol=1e-8)
```

**Failing input**: `n=1`

## Reproducing the Bug

```python
import scipy.sparse as sp
import scipy.sparse.linalg as sla

A = sp.csc_array([[2.0]])
Ainv = sla.inv(A)

print(f"Input type: {type(A)}")
print(f"Output type: {type(Ainv)}")
print(f"Output value: {Ainv}")
```

Output:
```
Input type: <class 'scipy.sparse._csc.csc_array'>
Output type: <class 'numpy.ndarray'>
Output value: [0.5]
```

## Why This Is A Bug

The function documentation explicitly states:

```
Returns
-------
Ainv : (M, M) sparse arrays
    inverse of `A`
```

For a 1x1 matrix, the function returns a 1D `numpy.ndarray` instead of a sparse array. This breaks:
1. Type consistency - the return type varies based on input size
2. The documented API contract
3. Composability - `inv(inv(A))` fails for 1x1 matrices because the second `inv` call receives an ndarray and raises `TypeError: Input must be a sparse arrays`

## Fix

The root cause is that `spsolve` returns a 1D ndarray when solving with a (1,1) matrix, treating it as a vector. The fix should ensure `inv` always returns a sparse array:

```diff
--- a/scipy/sparse/linalg/_matfuncs.py
+++ b/scipy/sparse/linalg/_matfuncs.py
@@ -69,5 +69,10 @@ def inv(A):
     # Use sparse direct solver to solve "AX = I" accurately
     I = _ident_like(A)
     Ainv = spsolve(A, I)
+
+    # spsolve returns ndarray for 1x1 matrices; ensure consistent sparse return
+    if not (issparse(Ainv) or is_pydata_spmatrix(Ainv)):
+        Ainv = type(A)(Ainv.reshape(A.shape))
+
     return Ainv
```