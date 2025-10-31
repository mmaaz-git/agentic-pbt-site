# Bug Report: scipy.sparse.linalg.inv Inconsistent Return Type for 1x1 Matrices

**Target**: `scipy.sparse.linalg.inv`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.sparse.linalg.inv` returns a dense numpy ndarray for 1x1 matrices instead of a sparse matrix, violating its documented contract that explicitly promises to return sparse arrays regardless of matrix size.

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

# Run the test
if __name__ == "__main__":
    test_inv_returns_sparse()
```

<details>

<summary>
**Failing input**: `size=1, value=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 18, in <module>
    test_inv_returns_sparse()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 7, in test_inv_returns_sparse
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 14, in test_inv_returns_sparse
    assert sparse.issparse(A_inv), f"inv should return sparse array, got {type(A_inv)}"
           ~~~~~~~~~~~~~~~^^^^^^^
AssertionError: inv should return sparse array, got <class 'numpy.ndarray'>
Falsifying example: test_inv_returns_sparse(
    size=1,
    value=1.0,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

# Create sparse diagonal matrices of different sizes
A1 = sparse.diags([2.0], offsets=0, format='csr')
A2 = sparse.diags([2.0, 3.0], offsets=0, format='csr')
A3 = sparse.diags([1.0, 2.0, 3.0], offsets=0, format='csr')

# Compute the inverse of each matrix
inv1 = linalg.inv(A1)
inv2 = linalg.inv(A2)
inv3 = linalg.inv(A3)

# Print the types
print("1x1 matrix inverse type:", type(inv1))
print("2x2 matrix inverse type:", type(inv2))
print("3x3 matrix inverse type:", type(inv3))

# Check if they are sparse
print("\n1x1 is sparse:", sparse.issparse(inv1))
print("2x2 is sparse:", sparse.issparse(inv2))
print("3x3 is sparse:", sparse.issparse(inv3))

# Print the actual values
print("\n1x1 inverse value:", inv1)
print("2x2 inverse diagonal:", inv2.diagonal())
print("3x3 inverse diagonal:", inv3.diagonal())

# Demonstrate the error this can cause
print("\nTrying to call .toarray() on each result:")
try:
    print("1x1 toarray:", inv1.toarray())
except AttributeError as e:
    print("1x1 toarray failed:", e)

try:
    print("2x2 toarray shape:", inv2.toarray().shape)
except AttributeError as e:
    print("2x2 toarray failed:", e)

try:
    print("3x3 toarray shape:", inv3.toarray().shape)
except AttributeError as e:
    print("3x3 toarray failed:", e)
```

<details>

<summary>
AttributeError when calling .toarray() on 1x1 matrix inverse
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_dsolve/linsolve.py:606: SparseEfficiencyWarning: splu converted its input to CSC format
  return splu(A).solve
/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_matfuncs.py:76: SparseEfficiencyWarning: spsolve is more efficient when sparse b is in the CSC matrix format
  Ainv = spsolve(A, I)
1x1 matrix inverse type: <class 'numpy.ndarray'>
2x2 matrix inverse type: <class 'scipy.sparse._csr.csr_matrix'>
3x3 matrix inverse type: <class 'scipy.sparse._csr.csr_matrix'>

1x1 is sparse: False
2x2 is sparse: True
3x3 is sparse: True

1x1 inverse value: [0.5]
2x2 inverse diagonal: [0.5        0.33333333]
3x3 inverse diagonal: [1.         0.5        0.33333333]

Trying to call .toarray() on each result:
1x1 toarray failed: 'numpy.ndarray' object has no attribute 'toarray'
2x2 toarray shape: (2, 2)
3x3 toarray shape: (3, 3)
```
</details>

## Why This Is A Bug

The function `scipy.sparse.linalg.inv` explicitly documents in its docstring (line 42 in `/scipy/sparse/linalg/_matfuncs.py`) that it:

**Returns:**
- **Ainv : (M, M) sparse arrays** - inverse of `A`

The documentation makes an unambiguous promise with no exceptions mentioned for any matrix size. However, the implementation violates this contract specifically for 1x1 matrices by returning a dense numpy.ndarray instead of a sparse matrix.

The root cause is that `inv()` internally calls `spsolve(A, I)` to compute the inverse. The `spsolve` function has logic (lines 291-292 in `linsolve.py`) that flattens/ravels the result when it determines the right-hand side is a "vector". For 1x1 matrices, the identity matrix is treated as a vector, causing `spsolve` to return a 1D numpy array instead of a sparse matrix.

This inconsistency causes several problems:
1. **API Contract Violation**: The documented return type is not honored
2. **Code Breakage**: User code that calls sparse-specific methods like `.toarray()`, `.tocsr()`, or `.nnz` will crash with AttributeError for 1x1 matrices but work fine for all other sizes
3. **Type Inconsistency**: Users must add special case handling for 1x1 matrices, complicating their code
4. **Principle of Least Surprise**: A function in `scipy.sparse.linalg` should consistently return sparse types

## Relevant Context

- The bug only affects 1x1 sparse matrices, which are admittedly rare in practical sparse linear algebra applications
- The mathematical results are correct (the inverse values are accurate)
- The issue stems from `spsolve` treating 1x1 matrices as vectors internally
- Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.inv.html
- Source code: https://github.com/scipy/scipy/blob/main/scipy/sparse/linalg/_matfuncs.py#L31-L77

## Proposed Fix

The fix should ensure the return value is always a sparse matrix as documented. This can be achieved by checking the return type and converting if necessary:

```diff
--- a/scipy/sparse/linalg/_matfuncs.py
+++ b/scipy/sparse/linalg/_matfuncs.py
@@ -73,7 +73,12 @@ def inv(A):

     # Use sparse direct solver to solve "AX = I" accurately
     I = _ident_like(A)
     Ainv = spsolve(A, I)
+
+    # Ensure return value is sparse for consistency with documentation
+    if not issparse(Ainv):
+        # Convert dense result to sparse matrix (happens for 1x1 matrices)
+        Ainv = A.__class__([[Ainv[0]]])
+
     return Ainv
```