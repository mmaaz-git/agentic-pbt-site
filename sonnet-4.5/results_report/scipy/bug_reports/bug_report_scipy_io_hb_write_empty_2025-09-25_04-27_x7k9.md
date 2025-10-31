# Bug Report: scipy.io.hb_write Crashes on Empty Sparse Matrices

**Target**: `scipy.io.hb_write`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.io.hb_write` crashes with a `ValueError` when attempting to write a sparse matrix that contains no non-zero elements. This is a valid edge case that should be handled gracefully.

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
```

**Failing input**: `rows=1, cols=1, sparsity=0.5` (which creates a matrix with all zeros)

## Reproducing the Bug

```python
import numpy as np
import scipy.io
from scipy.sparse import csc_array
import tempfile

A = csc_array(np.zeros((10, 10)))

with tempfile.NamedTemporaryFile(mode='wb', suffix='.hb', delete=False) as f:
    fname = f.name

scipy.io.hb_write(fname, A)
```

**Output:**
```
Traceback (most recent call last):
  ...
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/_harwell_boeing/hb.py", line 79, in from_data
    indices_fmt = IntFormat.from_number(np.max(indices+1))
                                        ^^^^^^^^^^^^^^^^^
ValueError: zero-size array to reduction operation maximum which has no identity
```

## Why This Is A Bug

1. **Valid input**: An empty sparse matrix (all zeros) is a perfectly valid data structure that should be writable to any matrix format.

2. **Unexpected crash**: The function crashes instead of handling the edge case gracefully or raising a meaningful error message.

3. **Inconsistent with other formats**: Other scipy.io functions like `mmwrite` handle empty matrices correctly:
   ```python
   >>> scipy.io.mmwrite('test.mtx', csc_array(np.zeros((5, 5))))
   # Works fine
   ```

4. **Location of bug**: The bug is in `scipy/io/_harwell_boeing/hb.py` line 79, where `np.max(indices+1)` is called on an empty array when there are no non-zero elements.

## Fix

Handle the edge case of empty indices array:

```diff
--- a/scipy/io/_harwell_boeing/hb.py
+++ b/scipy/io/_harwell_boeing/hb.py
@@ -76,7 +76,10 @@ class HBInfo:
         pointer = data.indptr
         indices = data.indices

-        indices_fmt = IntFormat.from_number(np.max(indices+1))
+        if len(indices) > 0:
+            indices_fmt = IntFormat.from_number(np.max(indices+1))
+        else:
+            indices_fmt = IntFormat.from_number(1)  # Minimal format for empty matrix
         pointer_fmt = IntFormat.from_number(np.max(pointer))

         # Assuming real for now
```