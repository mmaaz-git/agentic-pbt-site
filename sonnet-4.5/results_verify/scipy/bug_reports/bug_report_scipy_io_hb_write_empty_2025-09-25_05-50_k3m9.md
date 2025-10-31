# Bug Report: scipy.io.hb_write Empty Sparse Matrix Crash

**Target**: `scipy.io.hb_write`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `hb_write` function crashes with a `ValueError` when attempting to write an empty sparse matrix (nnz=0), despite empty sparse matrices being valid and representing all-zero matrices.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import tempfile
import numpy as np
from scipy.io import hb_write, hb_read
from scipy.sparse import coo_array

@given(st.integers(min_value=1, max_value=20),
       st.integers(min_value=1, max_value=20),
       st.integers(min_value=0, max_value=50))
def test_hb_write_hb_read_roundtrip(rows, cols, nnz):
    assume(nnz <= rows * cols)

    row_indices = np.random.randint(0, rows, size=nnz)
    col_indices = np.random.randint(0, cols, size=nnz)
    data = np.random.rand(nnz)

    sparse_matrix = coo_array((data, (row_indices, col_indices)), shape=(rows, cols))

    with tempfile.NamedTemporaryFile(suffix='.hb', delete=False, mode='w') as f:
        filepath = f.name

    try:
        hb_write(filepath, sparse_matrix)
        result = hb_read(filepath, spmatrix=False)

        assert result.shape == sparse_matrix.shape
        result_dense = result.toarray()
        expected_dense = sparse_matrix.toarray()
        assert np.allclose(result_dense, expected_dense)
    finally:
        import os
        if os.path.exists(filepath):
            os.unlink(filepath)
```

**Failing input**: `rows=1, cols=1, nnz=0`

## Reproducing the Bug

```python
import tempfile
from scipy.io import hb_write
from scipy.sparse import coo_array

sparse_matrix = coo_array(([], ([], [])), shape=(5, 5))

with tempfile.NamedTemporaryFile(suffix='.hb', delete=False, mode='w') as f:
    filepath = f.name

hb_write(filepath, sparse_matrix)
```

Output:
```
ValueError: zero-size array to reduction operation maximum which has no identity
```

## Why This Is A Bug

Empty sparse matrices are valid data structures representing all-zero matrices. The `hb_write` function should handle this edge case gracefully instead of crashing. The crash occurs because `np.max()` is called on empty arrays without providing a default value or initial parameter.

## Fix

```diff
--- a/scipy/io/_harwell_boeing/hb.py
+++ b/scipy/io/_harwell_boeing/hb.py
@@ -75,8 +75,13 @@ class HBInfo:
         if fmt is None:
             # +1 because HB use one-based indexing (Fortran), and we will write
             # the indices /pointer as such
-            pointer_fmt = IntFormat.from_number(np.max(pointer+1))
-            indices_fmt = IntFormat.from_number(np.max(indices+1))
+            if pointer.size > 0:
+                pointer_fmt = IntFormat.from_number(np.max(pointer+1))
+            else:
+                pointer_fmt = IntFormat.from_number(1)
+            if indices.size > 0:
+                indices_fmt = IntFormat.from_number(np.max(indices+1))
+            else:
+                indices_fmt = IntFormat.from_number(1)

             if values.dtype.kind in np.typecodes["AllFloat"]:
-                values_fmt = ExpFormat.from_number(-np.max(np.abs(values)))
+                if values.size > 0:
+                    values_fmt = ExpFormat.from_number(-np.max(np.abs(values)))
+                else:
+                    values_fmt = ExpFormat.from_number(0.0)
             elif values.dtype.kind in np.typecodes["AllInteger"]:
-                values_fmt = IntFormat.from_number(-np.max(np.abs(values)))
+                if values.size > 0:
+                    values_fmt = IntFormat.from_number(-np.max(np.abs(values)))
+                else:
+                    values_fmt = IntFormat.from_number(0)
             else:
                 message = f"type {values.dtype.kind} not implemented yet"
                 raise NotImplementedError(message)
```