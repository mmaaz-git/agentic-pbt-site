# Bug Report: scipy.sparse.linalg.inv Inconsistent Return Type

**Target**: `scipy.sparse.linalg.inv`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.sparse.linalg.inv` returns a dense numpy ndarray for 1x1 matrices instead of a sparse array, violating its documented contract that it should return sparse arrays.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

@settings(max_examples=500)
@given(
    size=st.integers(min_value=1, max_value=5),
    value=st.floats(allow_nan=False, allow_infinity=False, min_value=0.1, max_value=10)
)
def test_inv_returns_sparse(size, value):
    A = sparse.diags([value] * size, offsets=0, format='csr')
    A_inv = splinalg.inv(A)
    assert sparse.issparse(A_inv), f"inv should return sparse array, got {type(A_inv)}"
```

**Failing input**: `size=1, value=1.0` (or any value with size=1)

## Reproducing the Bug

```python
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

A1 = sparse.diags([2.0], offsets=0, format='csr')
A2 = sparse.diags([2.0, 3.0], offsets=0, format='csr')

inv1 = linalg.inv(A1)
inv2 = linalg.inv(A2)

print("1x1 matrix inverse type:", type(inv1))
print("2x2 matrix inverse type:", type(inv2))

print("\n1x1 is sparse:", sparse.issparse(inv1))
print("2x2 is sparse:", sparse.issparse(inv2))
```

**Output:**
```
1x1 matrix inverse type: <class 'numpy.ndarray'>
2x2 matrix inverse type: <class 'scipy.sparse._csc.csc_array'>

1x1 is sparse: False
2x2 is sparse: True
```

## Why This Is A Bug

The documentation for `scipy.sparse.linalg.inv` explicitly states in the Returns section:

> **Ainv**: (M, M) sparse arrays
> Inverse of the sparse matrix A

The function returns a sparse array for matrices of size 2x2 and larger, but returns a dense numpy ndarray for 1x1 matrices. This inconsistency:

1. Violates the API contract documented in the function's docstring
2. Breaks user code that expects sparse output (e.g., calling `.toarray()` on the result)
3. Is inconsistent behavior based on input size
4. Can cause performance issues if users expect sparsity to be preserved

## Fix

The function should consistently return sparse arrays regardless of matrix size. Looking at the source in `scipy/sparse/linalg/_matfuncs.py`, the likely issue is in the `inv` function which internally uses `spsolve`, and `spsolve` may return dense arrays for 1x1 matrices.

The fix should ensure the return value is always converted to a sparse array:

```diff
--- a/scipy/sparse/linalg/_matfuncs.py
+++ b/scipy/sparse/linalg/_matfuncs.py
@@ -73,7 +73,10 @@ def inv(A):
     I = eye(A.shape[0], format=A.format)
     Ainv = spsolve(A, I)
-    return Ainv
+    if not issparse(Ainv):
+        # Ensure return value is sparse for consistency
+        Ainv = A.__class__(Ainv)
+    return Ainv
```