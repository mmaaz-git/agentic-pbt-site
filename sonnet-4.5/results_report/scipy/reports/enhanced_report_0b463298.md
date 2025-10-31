# Bug Report: scipy.io.hb_write Crashes on Empty Sparse Matrices

**Target**: `scipy.io.hb_write`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.io.hb_write` crashes with a `ValueError` when attempting to write an empty sparse matrix (a matrix with zero non-zero elements) to a Harwell-Boeing format file.

## Property-Based Test

```python
import numpy as np
import scipy.io
import scipy.sparse
import tempfile
import os
from hypothesis import given, strategies as st, settings


@given(
    st.integers(1, 10),
    st.integers(1, 10),
    st.floats(allow_nan=False, allow_infinity=False, min_value=0.01, max_value=1.0)
)
@settings(max_examples=100)
def test_harwell_boeing_roundtrip(rows, cols, density):
    arr = scipy.sparse.random(rows, cols, density=density, format='csr', dtype=np.float64)

    with tempfile.NamedTemporaryFile(suffix='.hb', delete=False) as f:
        filename = f.name

    try:
        scipy.io.hb_write(filename, arr)
        loaded = scipy.io.hb_read(filename, spmatrix=False)

        if scipy.sparse.issparse(loaded):
            loaded_dense = loaded.toarray()
        else:
            loaded_dense = loaded

        arr_dense = arr.toarray()

        assert loaded_dense.shape == arr_dense.shape
        assert np.allclose(loaded_dense, arr_dense, rtol=1e-9, atol=1e-14)
    finally:
        if os.path.exists(filename):
            os.unlink(filename)


# Run the test
if __name__ == "__main__":
    test_harwell_boeing_roundtrip()
```

<details>

<summary>
**Failing input**: `test_harwell_boeing_roundtrip(rows=1, cols=1, density=0.5)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 41, in <module>
    test_harwell_boeing_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 10, in test_harwell_boeing_roundtrip
    st.integers(1, 10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 22, in test_harwell_boeing_roundtrip
    scipy.io.hb_write(filename, arr)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/_harwell_boeing/hb.py", line 561, in hb_write
    hb_info = HBInfo.from_data(m)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/_harwell_boeing/hb.py", line 79, in from_data
    indices_fmt = IntFormat.from_number(np.max(indices+1))
                                        ~~~~~~^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 3163, in max
    return _wrapreduction(a, np.maximum, 'max', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 86, in _wrapreduction
    return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: zero-size array to reduction operation maximum which has no identity
Falsifying example: test_harwell_boeing_roundtrip(
    rows=1,
    cols=1,
    density=0.5,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_compressed.py:47
        /home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_compressed.py:50
        /home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_compressed.py:51
        /home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_compressed.py:52
        /home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_compressed.py:102
        (and 5 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import scipy.sparse
import scipy.io
import numpy as np

# Create an empty sparse matrix
empty_sparse = scipy.sparse.csr_array((3, 3), dtype=np.float64)

print(f"Empty sparse matrix shape: {empty_sparse.shape}")
print(f"Empty sparse matrix nnz: {empty_sparse.nnz}")
print(f"Empty sparse matrix data: {empty_sparse.data}")
print(f"Empty sparse matrix indices: {empty_sparse.indices}")
print(f"Empty sparse matrix indptr: {empty_sparse.indptr}")
print()

scipy.io.hb_write("test_empty.hb", empty_sparse)
```

<details>

<summary>
ValueError: zero-size array to reduction operation maximum which has no identity
</summary>
```
Empty sparse matrix shape: (3, 3)
Empty sparse matrix nnz: 0
Empty sparse matrix data: []
Empty sparse matrix indices: []
Empty sparse matrix indptr: [0 0 0 0]

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/repo.py", line 15, in <module>
    scipy.io.hb_write("test_empty.hb", empty_sparse)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/_harwell_boeing/hb.py", line 561, in hb_write
    hb_info = HBInfo.from_data(m)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/_harwell_boeing/hb.py", line 79, in from_data
    indices_fmt = IntFormat.from_number(np.max(indices+1))
                                        ~~~~~~^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 3163, in max
    return _wrapreduction(a, np.maximum, 'max', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 86, in _wrapreduction
    return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: zero-size array to reduction operation maximum which has no identity
```
</details>

## Why This Is A Bug

Empty sparse matrices are mathematically valid objects that represent matrices where all elements are zero. These occur naturally in numerical computations through operations like:
- Filtering or masking that removes all non-zero elements
- Initializing matrices before populating them with data
- Results from subtraction of identical sparse matrices
- Intermediate results in algorithms that progressively sparsify matrices

The scipy.io.hb_write documentation states it writes "sparse array or matrix" without any restriction on the number of non-zero elements. The function signature and documentation imply that any valid sparse matrix should be writable to the Harwell-Boeing format.

The crash occurs because the code in `HBInfo.from_data` attempts to compute `np.max(indices+1)` on line 79 when `indices` is an empty array, and similarly would fail on lines 82/84 when computing `np.max(np.abs(values))` for an empty values array. NumPy's `np.max()` function raises a ValueError when called on empty arrays because there is no identity element for the maximum operation.

## Relevant Context

The Harwell-Boeing format is a standard exchange format for sparse matrices commonly used in scientific computing. The format specification itself can represent matrices with zero non-zero elements by setting the appropriate metadata fields.

The bug manifests in two places in the code:
1. Line 79: `indices_fmt = IntFormat.from_number(np.max(indices+1))` - fails on empty indices array
2. Lines 82/84: Format determination for values - would also fail on empty values array

Note that `scipy.sparse.random()` with small dimensions can produce empty matrices even with non-zero density parameters due to randomization, making this bug more likely to be encountered than it might initially appear.

## Proposed Fix

```diff
--- a/scipy/io/_harwell_boeing/hb.py
+++ b/scipy/io/_harwell_boeing/hb.py
@@ -76,12 +76,20 @@ class HBInfo:
         if fmt is None:
             # +1 because HB use one-based indexing (Fortran), and we will write
             # the indices /pointer as such
             pointer_fmt = IntFormat.from_number(np.max(pointer+1))
-            indices_fmt = IntFormat.from_number(np.max(indices+1))
+            if len(indices) > 0:
+                indices_fmt = IntFormat.from_number(np.max(indices+1))
+            else:
+                # For empty matrices, use format based on matrix dimensions
+                indices_fmt = IntFormat.from_number(nrows)

             if values.dtype.kind in np.typecodes["AllFloat"]:
-                values_fmt = ExpFormat.from_number(-np.max(np.abs(values)))
+                if len(values) > 0:
+                    values_fmt = ExpFormat.from_number(-np.max(np.abs(values)))
+                else:
+                    values_fmt = ExpFormat.from_number(0.0)
             elif values.dtype.kind in np.typecodes["AllInteger"]:
-                values_fmt = IntFormat.from_number(-np.max(np.abs(values)))
+                if len(values) > 0:
+                    values_fmt = IntFormat.from_number(-np.max(np.abs(values)))
+                else:
+                    values_fmt = IntFormat.from_number(0)
             else:
                 message = f"type {values.dtype.kind} not implemented yet"
```