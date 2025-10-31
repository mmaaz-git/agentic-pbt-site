# Bug Report: scipy.io.hb_write Empty Sparse Matrix Crash

**Target**: `scipy.io.hb_write`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.io.hb_write` crashes with a ValueError when attempting to write a sparse matrix with zero non-zero elements, instead of handling this valid edge case gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import scipy.sparse
import scipy.io
import tempfile
import os

@given(st.integers(min_value=1, max_value=10))
def test_hb_write_handles_empty_sparse_matrix(n):
    """Property: hb_write should handle empty sparse matrices without crashing"""
    empty_matrix = scipy.sparse.csr_matrix((n, n))

    fd, temp_path = tempfile.mkstemp(suffix='.hb')
    try:
        os.close(fd)
        scipy.io.hb_write(temp_path, empty_matrix)
        # Should not crash
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
```

**Failing input**: `n=1` (or any size empty sparse matrix)

## Reproducing the Bug

```python
import scipy.sparse
import scipy.io
import tempfile

empty_matrix = scipy.sparse.csr_matrix((1, 1))
print(f"Matrix shape: {empty_matrix.shape}")
print(f"Non-zero elements: {empty_matrix.nnz}")

fd, path = tempfile.mkstemp(suffix='.hb')
try:
    import os
    os.close(fd)
    scipy.io.hb_write(path, empty_matrix)
except ValueError as e:
    print(f"Error: {e}")
finally:
    import os
    if os.path.exists(path):
        os.unlink(path)
```

## Why This Is A Bug

Empty sparse matrices (matrices with zero non-zero elements) are valid data structures that should be supported by I/O functions. Users may legitimately need to save/load such matrices during computation (e.g., intermediate results, initialization states, or sparse matrix algorithms that produce empty results).

The crash occurs in `scipy/io/_harwell_boeing/hb.py` at line 79 in `HBInfo.from_data`:

```python
indices_fmt = IntFormat.from_number(np.max(indices+1))
```

When the sparse matrix has no non-zero elements, `indices` is an empty array `[]`, causing `np.max()` to raise: `ValueError: zero-size array to reduction operation maximum which has no identity`.

## Fix

```diff
--- a/scipy/io/_harwell_boeing/hb.py
+++ b/scipy/io/_harwell_boeing/hb.py
@@ -76,8 +76,11 @@ class HBInfo:
         if fmt is None:
             # +1 because HB use one-based indexing (Fortran), and we will write
             # the indices /pointer as such
-            pointer_fmt = IntFormat.from_number(np.max(pointer+1))
-            indices_fmt = IntFormat.from_number(np.max(indices+1))
+            if pointer.size > 0:
+                pointer_fmt = IntFormat.from_number(np.max(pointer+1))
+            else:
+                pointer_fmt = IntFormat.from_number(1)  # minimum format
+            indices_fmt = IntFormat.from_number(np.max(indices+1) if indices.size > 0 else 1)

             if values.dtype.kind in np.typecodes["AllFloat"]:
                 values_fmt = ExpFormat.from_number(-np.max(np.abs(values)))
```

Alternatively, the function could raise a more informative error message explaining that empty sparse matrices are not supported by the Harwell-Boeing format.