# Bug Report: scipy.sparse.linalg.inv Returns Wrong Type and Shape for 1×1 Sparse Matrices

**Target**: `scipy.sparse.linalg.inv`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `scipy.sparse.linalg.inv` function returns a 1D numpy array with shape (1,) instead of a 2D sparse matrix with shape (1, 1) when inverting 1×1 sparse matrices, violating its documented contract.

## Property-Based Test

```python
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from hypothesis import given, strategies as st, settings


@given(n=st.integers(min_value=1, max_value=10))
@settings(max_examples=20)
def test_inv_always_returns_sparse_matrix(n):
    A = sp.diags([2.0 + i for i in range(n)], format='csr')

    A_inv = spl.inv(A)

    assert sp.issparse(A_inv), f"inv() returned {type(A_inv)} instead of sparse matrix for {n}x{n} input"
    assert A_inv.shape == (n, n), f"inv() returned shape {A_inv.shape} instead of ({n}, {n})"


if __name__ == "__main__":
    test_inv_always_returns_sparse_matrix()
```

<details>

<summary>
**Failing input**: `n=1`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_dsolve/linsolve.py:606: SparseEfficiencyWarning: splu converted its input to CSC format
  return splu(A).solve
/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_matfuncs.py:76: SparseEfficiencyWarning: spsolve is more efficient when sparse b is in the CSC matrix format
  Ainv = spsolve(A, I)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 18, in <module>
    test_inv_always_returns_sparse_matrix()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 7, in test_inv_always_returns_sparse_matrix
    @settings(max_examples=20)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 13, in test_inv_always_returns_sparse_matrix
    assert sp.issparse(A_inv), f"inv() returned {type(A_inv)} instead of sparse matrix for {n}x{n} input"
           ~~~~~~~~~~~^^^^^^^
AssertionError: inv() returned <class 'numpy.ndarray'> instead of sparse matrix for 1x1 input
Falsifying example: test_inv_always_returns_sparse_matrix(
    n=1,
)
```
</details>

## Reproducing the Bug

```python
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numpy as np

# Test case 1: 1x1 sparse matrix
print("=" * 60)
print("Test case 1: 1x1 sparse matrix")
print("=" * 60)
A = sp.csr_matrix([[2.0]])
print(f"Input matrix A:")
print(f"  Type: {type(A)}")
print(f"  Shape: {A.shape}")
print(f"  Value: {A.toarray()}")
print(f"  Is sparse: {sp.issparse(A)}")

A_inv = spl.inv(A)
print(f"\nResult of inv(A):")
print(f"  Type: {type(A_inv)}")
print(f"  Shape: {A_inv.shape}")
print(f"  Is sparse: {sp.issparse(A_inv)}")
print(f"  Value: {A_inv}")

# Test case 2: 2x2 sparse matrix for comparison
print("\n" + "=" * 60)
print("Test case 2: 2x2 sparse matrix (for comparison)")
print("=" * 60)
B = sp.csr_matrix([[2.0, 0.0], [0.0, 3.0]])
print(f"Input matrix B:")
print(f"  Type: {type(B)}")
print(f"  Shape: {B.shape}")
print(f"  Value:\n{B.toarray()}")
print(f"  Is sparse: {sp.issparse(B)}")

B_inv = spl.inv(B)
print(f"\nResult of inv(B):")
print(f"  Type: {type(B_inv)}")
print(f"  Shape: {B_inv.shape}")
print(f"  Is sparse: {sp.issparse(B_inv)}")
print(f"  Value:\n{B_inv.toarray()}")

# Test case 3: Demonstrating the issue is in spsolve
print("\n" + "=" * 60)
print("Test case 3: Direct spsolve call with 1x1 identity")
print("=" * 60)
A = sp.csr_matrix([[2.0]])
I = sp.eye(1, format='csr')
print(f"Identity matrix I:")
print(f"  Type: {type(I)}")
print(f"  Shape: {I.shape}")
print(f"  Value: {I.toarray()}")

result = spl.spsolve(A, I)
print(f"\nResult of spsolve(A, I):")
print(f"  Type: {type(result)}")
print(f"  Shape: {result.shape}")
print(f"  Is sparse: {sp.issparse(result)}")
print(f"  Value: {result}")

# Expected behavior
print("\n" + "=" * 60)
print("EXPECTED BEHAVIOR for 1x1 case:")
print("=" * 60)
print("  Type: Should be <class 'scipy.sparse._csr.csr_matrix'>")
print("  Shape: Should be (1, 1)")
print("  Is sparse: Should be True")
print("  Value: Should be [[0.5]] as a sparse matrix")
```

<details>

<summary>
AssertionError: inv() returns numpy.ndarray instead of sparse matrix for 1x1 input
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_dsolve/linsolve.py:606: SparseEfficiencyWarning: splu converted its input to CSC format
  return splu(A).solve
/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_matfuncs.py:76: SparseEfficiencyWarning: spsolve is more efficient when sparse b is in the CSC matrix format
  Ainv = spsolve(A, I)
============================================================
Test case 1: 1x1 sparse matrix
============================================================
Input matrix A:
  Type: <class 'scipy.sparse._csr.csr_matrix'>
  Shape: (1, 1)
  Value: [[2.]]
  Is sparse: True

Result of inv(A):
  Type: <class 'numpy.ndarray'>
  Shape: (1,)
  Is sparse: False
  Value: [0.5]

============================================================
Test case 2: 2x2 sparse matrix (for comparison)
============================================================
Input matrix B:
  Type: <class 'scipy.sparse._csr.csr_matrix'>
  Shape: (2, 2)
  Value:
[[2. 0.]
 [0. 3.]]
  Is sparse: True

Result of inv(B):
  Type: <class 'scipy.sparse._csr.csr_matrix'>
  Shape: (2, 2)
  Is sparse: True
  Value:
[[0.5        0.        ]
 [0.         0.33333333]]

============================================================
Test case 3: Direct spsolve call with 1x1 identity
============================================================
Identity matrix I:
  Type: <class 'scipy.sparse._csr.csr_matrix'>
  Shape: (1, 1)
  Value: [[1.]]

Result of spsolve(A, I):
  Type: <class 'numpy.ndarray'>
  Shape: (1,)
  Is sparse: False
  Value: [0.5]

============================================================
EXPECTED BEHAVIOR for 1x1 case:
============================================================
  Type: Should be <class 'scipy.sparse._csr.csr_matrix'>
  Shape: Should be (1, 1)
  Is sparse: Should be True
  Value: Should be [[0.5]] as a sparse matrix
```
</details>

## Why This Is A Bug

This violates the documented contract of `scipy.sparse.linalg.inv`. The function's documentation explicitly states that it returns "(M, M) sparse arrays" with no exceptions for special cases. However, when given a 1×1 sparse matrix as input:

1. **Wrong type**: Returns `numpy.ndarray` instead of a sparse array (violates "sparse arrays" promise)
2. **Wrong shape**: Returns shape `(1,)` instead of `(1, 1)` (violates "(M, M)" promise)
3. **Inconsistent behavior**: Works correctly for all n×n matrices where n≥2, but fails for n=1

The bug originates in `scipy.sparse.linalg.spsolve`, which `inv()` calls internally. The `inv()` function computes the inverse by solving AX = I, where I is the identity matrix. When `spsolve` receives a 1×1 sparse matrix as the right-hand side (b parameter), it incorrectly flattens the result to a 1D array instead of preserving the 2D matrix structure.

This inconsistency forces users to add special-case handling for 1×1 matrices in their code, making generic matrix operations unnecessarily complex and error-prone.

## Relevant Context

The `scipy.sparse.linalg.inv` function (located in scipy/sparse/linalg/_matfuncs.py) has a simple implementation:

```python
def inv(A):
    # Check input
    if not (issparse(A) or is_pydata_spmatrix(A)):
        raise TypeError('Input must be a sparse arrays')

    # Use sparse direct solver to solve "AX = I" accurately
    I = _ident_like(A)
    Ainv = spsolve(A, I)
    return Ainv
```

The issue is that `spsolve` behaves inconsistently:
- For n×n sparse RHS where n≥2: Returns n×n sparse matrix (correct)
- For 1×1 sparse RHS: Returns 1D numpy array with shape (1,) (incorrect)

Documentation links:
- inv documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.inv.html
- spsolve documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html

## Proposed Fix

The fix should be applied in `scipy.sparse.linalg.inv` to handle the 1×1 edge case explicitly:

```diff
def inv(A):
    """
    Compute the inverse of a sparse arrays

    Parameters
    ----------
    A : (M, M) sparse arrays
        square matrix to be inverted

    Returns
    -------
    Ainv : (M, M) sparse arrays
        inverse of `A`

    ...
    """
    # Check input
    if not (issparse(A) or is_pydata_spmatrix(A)):
        raise TypeError('Input must be a sparse arrays')

    # Use sparse direct solver to solve "AX = I" accurately
    I = _ident_like(A)
    Ainv = spsolve(A, I)
+
+   # Fix for 1x1 matrices: spsolve returns 1D array instead of sparse matrix
+   if A.shape == (1, 1) and not issparse(Ainv):
+       # Convert 1D array back to sparse matrix of appropriate format
+       Ainv = A.__class__([[Ainv[0]]])
+
    return Ainv
```