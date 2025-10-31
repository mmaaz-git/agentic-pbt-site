# Bug Report: scipy.io.hb_write Crashes on Empty Sparse Matrices

**Target**: `scipy.io.hb_write`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.io.hb_write` crashes with a `ValueError` when attempting to write sparse matrices that contain no non-zero elements, which are valid mathematical objects that should be handled gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.io
from scipy.sparse import csc_array
import tempfile
import os


@given(st.integers(min_value=1, max_value=30),
       st.integers(min_value=1, max_value=30),
       st.floats(min_value=0.1, max_value=0.9))
@settings(max_examples=50)
def test_hb_write_read_roundtrip_sparse(rows, cols, sparsity):
    np.random.seed(42)
    A_dense = np.random.random((rows, cols))
    A_dense[A_dense < sparsity] = 0
    A = csc_array(A_dense)

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.hb', delete=False) as f:
        fname = f.name

    try:
        scipy.io.hb_write(fname, A)
        B = scipy.io.hb_read(fname)
        B_dense = B.toarray() if hasattr(B, 'toarray') else B
        A_dense_check = A.toarray() if hasattr(A, 'toarray') else A
        assert np.allclose(A_dense_check, B_dense, rtol=1e-10)
    finally:
        if os.path.exists(fname):
            os.remove(fname)

if __name__ == "__main__":
    test_hb_write_read_roundtrip_sparse()
```

<details>

<summary>
**Failing input**: `rows=1, cols=1, sparsity=0.5`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 33, in <module>
    test_hb_write_read_roundtrip_sparse()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 10, in test_hb_write_read_roundtrip_sparse
    st.integers(min_value=1, max_value=30),
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 23, in test_hb_write_read_roundtrip_sparse
    scipy.io.hb_write(fname, A)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^
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
Falsifying example: test_hb_write_read_roundtrip_sparse(
    rows=1,
    cols=1,
    sparsity=0.5,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.io
from scipy.sparse import csc_array
import tempfile

# Create a 10x10 sparse matrix with all zeros (empty sparse matrix)
A = csc_array(np.zeros((10, 10)))

# Create a temporary file
with tempfile.NamedTemporaryFile(mode='wb', suffix='.hb', delete=False) as f:
    fname = f.name

# Try to write the empty sparse matrix to HB format
scipy.io.hb_write(fname, A)
```

<details>

<summary>
ValueError: zero-size array to reduction operation maximum which has no identity
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/repo.py", line 14, in <module>
    scipy.io.hb_write(fname, A)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^
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

This is a legitimate bug that violates expected behavior for the following reasons:

1. **Empty sparse matrices are mathematically valid**: A sparse matrix with zero non-zero elements is a valid sparse matrix representation (essentially a zero matrix). These occur naturally in many algorithms during initialization, as results of masking operations, or as intermediate values during computations.

2. **The function accepts the input type**: The function signature accepts sparse arrays/matrices and doesn't document any restriction on minimum non-zero elements. The crash occurs deep in implementation details rather than during input validation.

3. **Inconsistent behavior with other scipy.io functions**: Other sparse matrix I/O functions in scipy handle empty matrices correctly. For example:
   ```python
   scipy.io.mmwrite('test.mtx', csc_array(np.zeros((5, 5))))  # Works fine
   ```

4. **Confusing error message**: The error "zero-size array to reduction operation maximum which has no identity" is a low-level NumPy implementation detail that doesn't help users understand that the issue is with an empty sparse matrix.

5. **Documentation gap**: Neither the scipy.io.hb_write documentation nor the Harwell-Boeing format specification explicitly states that empty sparse matrices are unsupported. The documentation states it supports "assembled, non-symmetric, real matrices" without any minimum non-zero requirement.

## Relevant Context

The bug occurs in `/home/npc/.local/lib/python3.13/site-packages/scipy/io/_harwell_boeing/hb.py` at line 79 in the `HBInfo.from_data` method. When a sparse matrix has no non-zero elements:

- The `indices` array (containing row indices of non-zero elements) is empty
- The code attempts `np.max(indices+1)` to determine the format width needed for indices
- NumPy's `max()` function cannot operate on an empty array without an identity element, causing the crash

The Harwell-Boeing format specification tracks the NNZERO field (number of non-zero entries) but doesn't explicitly forbid nnz=0. The format structure includes header information with matrix dimensions and storage for non-zero values. An empty sparse matrix could be represented with nnz=0 and empty arrays for indices and values.

Related documentation:
- SciPy hb_write: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.hb_write.html
- Harwell-Boeing format: https://math.nist.gov/MatrixMarket/formats.html#hb

## Proposed Fix

```diff
--- a/scipy/io/_harwell_boeing/hb.py
+++ b/scipy/io/_harwell_boeing/hb.py
@@ -76,8 +76,14 @@ class HBInfo:
         if fmt is None:
             # +1 because HB use one-based indexing (Fortran), and we will write
             # the indices /pointer as such
             pointer_fmt = IntFormat.from_number(np.max(pointer+1))
-            indices_fmt = IntFormat.from_number(np.max(indices+1))
+
+            # Handle empty sparse matrices (no non-zero elements)
+            if len(indices) > 0:
+                indices_fmt = IntFormat.from_number(np.max(indices+1))
+            else:
+                # Use minimal format for empty matrix
+                indices_fmt = IntFormat.from_number(1)

             if values.dtype.kind in np.typecodes["AllFloat"]:
                 values_fmt = ExpFormat.from_number(-np.max(np.abs(values)))
```