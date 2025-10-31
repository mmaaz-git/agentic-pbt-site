# Bug Report: numpy.matrix.__getitem__ Allows 3D Matrix Creation via np.newaxis

**Target**: `numpy.matrixlib.defmatrix.matrix.__getitem__`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `matrix.__getitem__` method violates the documented 2-dimensional constraint by allowing creation of 3-dimensional matrix objects through `np.newaxis` indexing, causing failures in matrix-specific operations.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st


@given(st.integers(2, 10), st.integers(2, 10))
def test_matrix_remains_2d_after_operations(rows, cols):
    m = np.matrix(np.random.randn(rows, cols))
    assert m.ndim == 2

    result = m[:, np.newaxis, :]
    assert result.ndim == 2, f"Expected 2D matrix, got {result.ndim}D with shape {result.shape}"
```

<details>

<summary>
**Failing input**: `rows=2, cols=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 15, in <module>
    test_matrix_remains_2d_after_operations()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 6, in test_matrix_remains_2d_after_operations
    def test_matrix_remains_2d_after_operations(rows, cols):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 11, in test_matrix_remains_2d_after_operations
    assert result.ndim == 2, f"Expected 2D matrix, got {result.ndim}D with shape {result.shape}"
           ^^^^^^^^^^^^^^^^
AssertionError: Expected 2D matrix, got 3D with shape (2, 1, 2)
Falsifying example: test_matrix_remains_2d_after_operations(
    # The test always failed when commented parts were varied together.
    rows=2,  # or any other generated value
    cols=2,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np

# Create a 2D matrix
m = np.matrix([[1, 2, 3], [4, 5, 6]])
print(f"Original: shape={m.shape}, ndim={m.ndim}, type={type(m)}")

# Use np.newaxis indexing to create a 3D matrix (this should not be allowed!)
result = m[:, np.newaxis, :]
print(f"After indexing: shape={result.shape}, ndim={result.ndim}, type={type(result)}")
print(f"Is matrix: {isinstance(result, np.matrix)}")

# Check that we indeed have a 3D matrix object
assert result.ndim == 3
assert isinstance(result, np.matrix)
print("\nBug confirmed: Created a 3D matrix object!")

# Try to compute the inverse of the 3D matrix (this should fail)
try:
    inverse = result.I
    print(f"Inverse computed: {inverse}")
except Exception as e:
    print(f"\nAttempting to compute inverse fails with: {type(e).__name__}: {e}")
```

<details>

<summary>
Output demonstrates creation of invalid 3D matrix and broken inverse operation
</summary>
```
Original: shape=(2, 3), ndim=2, type=<class 'numpy.matrix'>
After indexing: shape=(2, 1, 3), ndim=3, type=<class 'numpy.matrix'>
Is matrix: True

Bug confirmed: Created a 3D matrix object!

Attempting to compute inverse fails with: ValueError: too many values to unpack (expected 2)
```
</details>

## Why This Is A Bug

This violates the fundamental invariant of the `numpy.matrix` class that is explicitly documented and enforced elsewhere in the code:

1. **Documentation states** (line 80-82 in defmatrix.py): "A matrix is a specialized 2-D array that retains its 2-D nature through operations."

2. **Constructor enforces 2D constraint** (line 155-156): The `__new__` method explicitly raises `ValueError("matrix must be 2-dimensional")` when `ndim > 2`.

3. **`__array_finalize__` enforces 2D constraint** (lines 181-188): This method also enforces the 2D constraint, either reshaping or raising `ValueError("shape too large to be a matrix.")` for arrays with `ndim > 2`.

4. **But `__getitem__` does NOT enforce this** (lines 197-221): The method handles `ndim == 0` and `ndim == 1` cases but completely ignores `ndim > 2`, allowing 3D matrices to pass through unchecked.

This inconsistency creates invalid matrix objects that:
- Violate the class's documented contract
- Break matrix-specific operations (e.g., `.I` property for inverse computation fails with "too many values to unpack")
- Behave inconsistently with matrices created through the constructor

## Relevant Context

The numpy.matrix class includes a deprecation warning (PendingDeprecationWarning) indicating it may be removed in the future. However, while the class exists, it should maintain consistency in enforcing its documented invariants.

The bug specifically occurs when using `np.newaxis` (which is an alias for `None`) in indexing operations like `m[:, np.newaxis, :]`. This adds a new dimension to the array, but `__getitem__` doesn't check if this violates the 2D constraint.

Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/numpy/matrixlib/defmatrix.py`

## Proposed Fix

Add dimensionality checking in the `__getitem__` method after line 206, similar to the check in `__array_finalize__`:

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -205,6 +205,14 @@ class matrix(N.ndarray):
         if not isinstance(out, N.ndarray):
             return out

+        # Enforce 2D constraint for matrices
+        if out.ndim > 2:
+            newshape = tuple(x for x in out.shape if x > 1)
+            if len(newshape) > 2:
+                raise ValueError("indexing operation would create matrix with dimensionality > 2")
+            # Reshape to maintain 2D if possible
+            out.shape = newshape if len(newshape) == 2 else (1, newshape[0] if newshape else 1)
+
         if out.ndim == 0:
             return out[()]
         if out.ndim == 1:
```