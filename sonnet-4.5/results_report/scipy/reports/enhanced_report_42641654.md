# Bug Report: scipy.sparse.linalg.inv Returns ndarray for 1x1 Matrices Instead of Sparse Array

**Target**: `scipy.sparse.linalg.inv`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `scipy.sparse.linalg.inv` function returns a 1D `numpy.ndarray` instead of a sparse array when given a 1x1 sparse matrix, violating its documented contract that promises to return sparse arrays.

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

<details>

<summary>
**Failing input**: `n=1, random=RandomSeeder(0)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 21, in <module>
    test_inv_double_inversion()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 7, in test_inv_double_inversion
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 14, in test_inv_double_inversion
    Ainvinv = sla.inv(Ainv)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_matfuncs.py", line 72, in inv
    raise TypeError('Input must be a sparse arrays')
TypeError: Input must be a sparse arrays
Falsifying example: test_inv_double_inversion(
    n=1,
    random=RandomSeeder(0),
)
```
</details>

## Reproducing the Bug

```python
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy as np

# Test case demonstrating the bug
A = sp.csc_array([[2.0]])
print(f"Input A type: {type(A)}")
print(f"Input A shape: {A.shape}")
print(f"Input A value: {A.toarray()}")

# First inversion
Ainv = sla.inv(A)
print(f"\nFirst inv(A) type: {type(Ainv)}")
print(f"First inv(A) shape: {Ainv.shape if hasattr(Ainv, 'shape') else 'N/A'}")
print(f"First inv(A) value: {Ainv}")

# Try second inversion - this will fail
try:
    Ainvinv = sla.inv(Ainv)
    print(f"\nSecond inv(inv(A)) type: {type(Ainvinv)}")
    print(f"Second inv(inv(A)) value: {Ainvinv}")
except TypeError as e:
    print(f"\nError on second inversion: {e}")

# Show that for 2x2 matrices it works correctly
print("\n--- For comparison, 2x2 matrix behavior ---")
A2 = sp.csc_array([[2.0, 0.0], [0.0, 3.0]])
print(f"Input A2 type: {type(A2)}")
print(f"Input A2 shape: {A2.shape}")

A2inv = sla.inv(A2)
print(f"First inv(A2) type: {type(A2inv)}")
print(f"First inv(A2) shape: {A2inv.shape}")

A2invinv = sla.inv(A2inv)
print(f"Second inv(inv(A2)) type: {type(A2invinv)}")
print(f"Second inv(inv(A2)) successful - no error")
```

<details>

<summary>
TypeError occurs when attempting double inversion of 1x1 matrix
</summary>
```
Input A type: <class 'scipy.sparse._csc.csc_array'>
Input A shape: (1, 1)
Input A value: [[2.]]

First inv(A) type: <class 'numpy.ndarray'>
First inv(A) shape: (1,)
First inv(A) value: [0.5]

Error on second inversion: Input must be a sparse arrays

--- For comparison, 2x2 matrix behavior ---
Input A2 type: <class 'scipy.sparse._csc.csc_array'>
Input A2 shape: (2, 2)
First inv(A2) type: <class 'scipy.sparse._csc.csc_array'>
First inv(A2) shape: (2, 2)
Second inv(inv(A2)) type: <class 'scipy.sparse._csc.csc_array'>
Second inv(inv(A2)) successful - no error
```
</details>

## Why This Is A Bug

The `scipy.sparse.linalg.inv` function violates its documented API contract in multiple ways:

1. **Type Contract Violation**: The function's docstring explicitly states "Returns: Ainv : (M, M) sparse arrays - inverse of A" with no exceptions mentioned for 1x1 matrices. For 1x1 input, it returns a 1D numpy.ndarray instead of a sparse array.

2. **Inconsistent Return Types**: The function returns different types based on matrix size:
   - 1x1 matrices → numpy.ndarray (1D array with shape (1,))
   - 2x2+ matrices → scipy.sparse array (2D sparse array with shape (n,n))

3. **Breaks Composability**: Mathematical operations like inv(inv(A)) should work for all valid inputs. This fails specifically for 1x1 matrices because the second inv() call receives an ndarray and raises `TypeError: Input must be a sparse arrays`.

4. **Shape Inconsistency**: The function returns a 1D array with shape (1,) for 1x1 matrices instead of maintaining the 2D shape (1,1) like the input.

## Relevant Context

The root cause is in `/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_matfuncs.py:76` where `inv` directly returns the result of `spsolve(A, I)`. The `spsolve` function treats 1x1 matrices as a special case and returns a 1D ndarray instead of a sparse matrix when solving Ax=b where b is a (1,1) matrix.

This behavior appears to be an optimization in `spsolve` for the common case of solving with a vector RHS, but it incorrectly applies this optimization even when the RHS is a matrix (albeit a 1x1 one). The inv function doesn't account for this edge case.

Documentation link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.inv.html

## Proposed Fix

```diff
--- a/scipy/sparse/linalg/_matfuncs.py
+++ b/scipy/sparse/linalg/_matfuncs.py
@@ -73,5 +73,11 @@ def inv(A):

     # Use sparse direct solver to solve "AX = I" accurately
     I = _ident_like(A)
     Ainv = spsolve(A, I)
+
+    # spsolve returns a 1D ndarray for 1x1 matrices, but we need to ensure
+    # consistent sparse array return type as documented
+    if not (issparse(Ainv) or is_pydata_spmatrix(Ainv)):
+        # Convert back to the same sparse type as input, maintaining shape
+        Ainv = type(A)(Ainv.reshape(A.shape))
+
     return Ainv
```