# Bug Report: scipy.sparse.kron stores explicit zeros violating sparse matrix invariants

**Target**: `scipy.sparse.kron`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.sparse.kron` creates sparse matrices with explicitly stored zero elements, violating the fundamental sparse matrix invariant that only nonzero values should be stored and causing the `nnz` attribute to report incorrect counts.

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

# Run the test
test_kron_with_identity()
```

<details>

<summary>
**Failing input**: `m=2, n=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 21, in <module>
    test_kron_with_identity()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 5, in test_kron_with_identity
    st.integers(min_value=2, max_value=6),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 18, in test_kron_with_identity
    assert result.nnz == m * A.nnz
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_kron_with_identity(
    m=2,
    n=2,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_base.py:473
        /home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_bsr.py:31
        /home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_bsr.py:40
        /home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_bsr.py:85
        /home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_bsr.py:90
        (and 30 more with settings.verbosity >= verbose)
```
</details>

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

<details>

<summary>
Explicit zeros in sparse matrix output
</summary>
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
</details>

## Why This Is A Bug

Sparse matrices are designed to store only nonzero elements for memory and computational efficiency. The `scipy.sparse.kron` function violates this fundamental invariant by explicitly storing zero values in the sparse matrix data structure. Specifically:

1. **Violated Invariant**: Sparse matrices should only store nonzero elements. The presence of explicit zeros (values of 0.0 in the `data` array) breaks this core assumption.

2. **Incorrect nnz Attribute**: The `nnz` (number of nonzeros) attribute reports 4 when there are only 2 actual nonzero values in the matrix. This misleading information can cause issues in algorithms that rely on accurate sparsity information.

3. **Memory Waste**: In the example, 50% of the stored values are zeros, defeating the purpose of using sparse matrices for memory efficiency. For larger matrices with identity-like patterns, this can lead to significant memory overhead.

4. **Performance Degradation**: Operations on sparse matrices often iterate over the stored elements. Including explicit zeros increases the iteration count unnecessarily, degrading performance in subsequent operations.

5. **Unexpected Behavior**: Users expect sparse matrix operations to maintain sparsity invariants automatically. Having to manually call `eliminate_zeros()` after `kron` is unexpected and error-prone.

## Relevant Context

The bug occurs in the BSR (Block Sparse Row) format path of the `kron` implementation. When computing the Kronecker product with relatively dense matrices (when `2*B.nnz >= B.shape[0] * B.shape[1]`), the code path at line 567-571 in `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/sparse/_construct.py` creates block matrices that include explicit zeros from the identity matrix's off-diagonal positions.

The COO (Coordinate) format path (lines 573-602) may have similar issues when constructing the Kronecker product, as it also doesn't filter out zero products.

Documentation for `scipy.sparse.kron`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.kron.html
Source code: scipy/sparse/_construct.py:503-603

## Proposed Fix

The fix should eliminate explicit zeros from the result before returning. The simplest approach is to call `eliminate_zeros()` on the result:

```diff
--- a/scipy/sparse/_construct.py
+++ b/scipy/sparse/_construct.py
@@ -569,7 +569,9 @@ def kron(A, B, format=None):
         data = A.data.repeat(B.size).reshape(-1,B.shape[0],B.shape[1])
         data = data * B

-        return bsr_sparse((data,A.indices,A.indptr), shape=output_shape)
+        result = bsr_sparse((data,A.indices,A.indptr), shape=output_shape)
+        result.eliminate_zeros()
+        return result
     else:
         # use COO
         A = coo_sparse(A)
@@ -599,7 +601,9 @@ def kron(A, B, format=None):
         data = data.reshape(-1,B.nnz) * B.data
         data = data.reshape(-1)

-        return coo_sparse((data,(row,col)), shape=output_shape).asformat(format)
+        result = coo_sparse((data,(row,col)), shape=output_shape).asformat(format)
+        result.eliminate_zeros()
+        return result.asformat(format)


 def kronsum(A, B, format=None):
```