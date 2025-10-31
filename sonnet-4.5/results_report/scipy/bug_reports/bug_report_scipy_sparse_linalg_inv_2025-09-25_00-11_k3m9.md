# Bug Report: scipy.sparse.linalg.inv Returns Wrong Type for 1×1 Matrices

**Target**: `scipy.sparse.linalg.inv`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `scipy.sparse.linalg.inv` function returns a 1D numpy array instead of a sparse matrix when inverting 1×1 sparse matrices, violating its documented contract that it returns sparse arrays.

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
```

**Failing input**: `n=1`

## Reproducing the Bug

```python
import scipy.sparse as sp
import scipy.sparse.linalg as spl

A = sp.csr_matrix([[2.0]])
A_inv = spl.inv(A)

print(f"Type: {type(A_inv)}")
print(f"Shape: {A_inv.shape}")
print(f"Is sparse: {sp.issparse(A_inv)}")
```

**Output:**
```
Type: <class 'numpy.ndarray'>
Shape: (1,)
Is sparse: False
```

**Expected behavior:**
```
Type: <class 'scipy.sparse._csr.csr_matrix'>
Shape: (1, 1)
Is sparse: True
```

## Why This Is A Bug

The documentation for `scipy.sparse.linalg.inv` explicitly states:

```
Returns
-------
Ainv : (M, M) sparse arrays
    inverse of `A`
```

However, for 1×1 input matrices:
1. The function returns a `numpy.ndarray` instead of a sparse array
2. The return value has shape `(1,)` instead of `(1, 1)`

This is a **contract violation** - the function does not behave as documented.

### Root Cause

The bug originates in `scipy.sparse.linalg.spsolve`, which `inv()` calls internally:

```python
def inv(A):
    I = _ident_like(A)
    Ainv = spsolve(A, I)
    return Ainv
```

When `spsolve` receives a 1×1 sparse matrix as the right-hand side, it incorrectly treats it as a vector and returns a 1D numpy array of shape `(1,)` instead of a 2D sparse matrix of shape `(1, 1)`.

**spsolve behavior:**
- For n×n sparse RHS where n ≥ 2: correctly returns n×n sparse matrix
- For 1×1 sparse RHS: incorrectly returns 1D array of shape (1,)

## Fix

The fix should be applied in `spsolve` to ensure it preserves the 2D structure when the right-hand side is a 1×1 sparse matrix. A high-level approach:

1. In `spsolve`, detect when `b` is a sparse matrix with shape `(1, 1)`
2. Ensure the result maintains the 2D sparse structure
3. Return a sparse matrix of shape `(1, 1)` instead of flattening to shape `(1,)`

Alternatively, `inv()` could work around this by explicitly handling the 1×1 case:

```python
def inv(A):
    I = _ident_like(A)
    Ainv = spsolve(A, I)

    # Workaround for spsolve bug with 1x1 matrices
    if A.shape == (1, 1) and not issparse(Ainv):
        Ainv = A.__class__([[Ainv[0]]])

    return Ainv
```

However, the proper fix should be in `spsolve` to ensure consistent behavior across all matrix sizes.