# Bug Report: numpy.matrix.__getitem__ Allows 3D Matrix Creation

**Target**: `numpy.matrixlib.defmatrix.matrix.__getitem__`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `matrix.__getitem__` method allows creation of 3-dimensional matrix objects through `np.newaxis` indexing, violating the documented constraint that matrices must be 2-dimensional.

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

**Failing input**: Any 2D matrix with shape (rows, cols) where rows >= 2 and cols >= 2

## Reproducing the Bug

```python
import numpy as np

m = np.matrix([[1, 2, 3], [4, 5, 6]])
print(f"Original: shape={m.shape}, ndim={m.ndim}")

result = m[:, np.newaxis, :]
print(f"After indexing: shape={result.shape}, ndim={result.ndim}")
print(f"Is matrix: {isinstance(result, np.matrix)}")

assert result.ndim == 3
assert isinstance(result, np.matrix)
```

## Why This Is A Bug

The `matrix` class documentation explicitly states:
> "A matrix is a specialized 2-D array that retains its 2-D nature through operations."

The constructor enforces this by raising `ValueError("matrix must be 2-dimensional")` for arrays with `ndim > 2`. However, `__getitem__` bypasses this constraint when using `np.newaxis`, allowing 3D matrix objects to exist.

This violates a fundamental class invariant and can cause:
1. Confusion about matrix dimensionality
2. Unexpected failures in operations expecting 2D matrices (e.g., `m.I` fails with "too many values to unpack")
3. Inconsistent behavior between constructor and indexing

## Fix

The `__getitem__` method should check the dimensionality of the result and either raise an error or reshape to 2D when `ndim > 2`, consistent with `__array_finalize__`:

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -205,6 +205,12 @@ class matrix(N.ndarray):
         if not isinstance(out, N.ndarray):
             return out

+        if out.ndim > 2:
+            newshape = tuple(x for x in out.shape if x > 1)
+            if len(newshape) > 2:
+                raise ValueError("indexing operation would create >2D matrix")
+            out.shape = newshape if len(newshape) == 2 else (1, -1)
+
         if out.ndim == 0:
             return out[()]
         if out.ndim == 1:
```