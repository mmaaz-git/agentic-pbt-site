# Bug Report: scipy.sparse COO Matrix Transpose Loses Canonical Format Flag

**Target**: `scipy.sparse.coo_matrix.transpose()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `transpose()` method of `coo_matrix` does not preserve the `has_canonical_format` flag, even though transposing a matrix with no duplicate entries cannot introduce duplicates.

## Property-Based Test

```python
import numpy as np
import scipy.sparse as sp
from hypothesis import given, strategies as st, settings


@st.composite
def sparse_coo_matrix(draw):
    m = draw(st.integers(min_value=1, max_value=20))
    n = draw(st.integers(min_value=1, max_value=20))
    nnz = draw(st.integers(min_value=0, max_value=min(m * n, 50)))

    if nnz == 0:
        return sp.coo_matrix((m, n), dtype=np.float64)

    rows = draw(st.lists(st.integers(min_value=0, max_value=m-1), min_size=nnz, max_size=nnz))
    cols = draw(st.lists(st.integers(min_value=0, max_value=n-1), min_size=nnz, max_size=nnz))
    data = draw(st.lists(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False), min_size=nnz, max_size=nnz))

    return sp.coo_matrix((data, (rows, cols)), shape=(m, n))


@given(sparse_coo_matrix())
@settings(max_examples=100)
def test_transpose_preserves_canonical_format(A):
    A.sum_duplicates()
    assert A.has_canonical_format == True

    A_T = A.transpose()

    assert A_T.has_canonical_format == True
```

**Failing input**: Any `coo_matrix` with `has_canonical_format=True`

## Reproducing the Bug

```python
import scipy.sparse as sp

data = [1, 2]
rows = [0, 0]
cols = [0, 0]
A = sp.coo_matrix((data, (rows, cols)), shape=(2, 2))
A.sum_duplicates()

print(f"A.has_canonical_format = {A.has_canonical_format}")

A_T = A.transpose()
print(f"A_T.has_canonical_format = {A_T.has_canonical_format}")
```

Output:
```
A.has_canonical_format = True
A_T.has_canonical_format = False
```

## Why This Is A Bug

The `has_canonical_format` flag indicates that a COO matrix has no duplicate (row, col) entries. Transposing a matrix simply swaps rows and columns - it cannot introduce duplicates if none existed before. Therefore, if `A.has_canonical_format` is `True`, then `A.transpose().has_canonical_format` should also be `True`.

This violates the semantic invariant that transpose is a structure-preserving operation. The flag's incorrect value can cause unnecessary duplicate summation operations and mislead users about the matrix's internal state.

## Fix

The bug is in `/scipy/sparse/_coo.py` in the `transpose()` method. Currently, it creates a new COO matrix from `(data, coords)` tuple, which always sets `has_canonical_format = False` (line 63 of `_coo.py`).

```diff
--- a/scipy/sparse/_coo.py
+++ b/scipy/sparse/_coo.py
@@ -291,8 +291,11 @@ class _coo_base(_data_matrix, _minmax_mixin):

         permuted_shape = tuple(self._shape[i] for i in axes)
         permuted_coords = tuple(self.coords[i] for i in axes)
-        return self.__class__((self.data, permuted_coords),
-                              shape=permuted_shape, copy=copy)
+        result = self.__class__((self.data, permuted_coords),
+                                shape=permuted_shape, copy=copy)
+        if self.has_canonical_format:
+            result.has_canonical_format = True
+        return result

     transpose.__doc__ = _spbase.transpose.__doc__
```