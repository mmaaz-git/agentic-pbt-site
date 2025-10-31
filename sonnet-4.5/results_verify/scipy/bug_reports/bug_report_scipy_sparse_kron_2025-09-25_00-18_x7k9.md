# Bug Report: scipy.sparse.kron stores explicit zeros

**Target**: `scipy.sparse.kron`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.sparse.kron` creates sparse matrices with explicitly stored zero elements, violating the fundamental sparse matrix invariant that only nonzero values should be stored.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import scipy.sparse as sp

@given(
    st.integers(min_value=2, max_value=6),
    st.integers(min_value=2, max_value=6)
)
@settings(max_examples=30)
def test_kron_with_identity(m, n):
    """kron(A, I) should only store nonzero elements"""
    A = sp.random(m, n, density=0.5)
    I = sp.eye(m)

    result = sp.kron(A, I)

    # The number of nonzeros should be m times A's nonzeros
    # Each nonzero in A contributes m nonzeros from the diagonal of I
    assert result.nnz == m * A.nnz
```

**Failing input**: `m=2, A=sp.random(2, 2, density=0.5)` (or any sparse matrix)

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp

A = sp.csr_array([[0, 1]])
I = sp.eye(2)

result = sp.kron(A, I).tocsr()

print(f"Result:\n{result.toarray()}")
print(f"nnz: {result.nnz}")
print(f"Actual nonzeros: {np.count_nonzero(result.toarray())}")
print(f"data array: {result.data}")
print(f"Explicit zeros: {np.sum(result.data == 0)}")

result.eliminate_zeros()
print(f"After eliminate_zeros(), nnz: {result.nnz}")
```

Output:
```
Result:
[[0. 0. 1. 0.]
 [0. 0. 0. 1.]]
nnz: 4
Actual nonzeros: 2
data array: [1. 0. 0. 1.]
Explicit zeros: 2
After eliminate_zeros(), nnz: 2
```

## Why This Is A Bug

Sparse matrices should only store nonzero elements. The `nnz` attribute should reflect the actual number of nonzero values. When `kron(A, I)` creates the Kronecker product, it stores all elements of `a[i,j] * I` including the zeros from the identity matrix's off-diagonal positions. This:

1. Wastes memory by storing zeros
2. Makes `nnz` misleading
3. Can degrade performance in subsequent operations
4. Violates the sparse matrix invariant

## Fix

The fix should call `eliminate_zeros()` on the result before returning, or better yet, avoid creating explicit zeros during construction:

```diff
--- a/scipy/sparse/_construct.py
+++ b/scipy/sparse/_construct.py
@@ -XXX,X +XXX,X @@ def kron(A, B, format=None):
         ...
         result = _compressed_sparse_stack(...)
+        result.eliminate_zeros()
         return result
```

Alternatively, the construction logic should be modified to only store actual nonzero products when building the Kronecker product.