# Bug Report: scipy.io.hb_write Empty Sparse Matrix Crash

**Target**: `scipy.io.hb_write`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.io.hb_write` crashes with a `ValueError` when attempting to write an empty sparse matrix (one with no non-zero elements) to a Harwell-Boeing format file.

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
```

**Failing input**: A sparse matrix with zero non-zero elements, e.g., `scipy.sparse.csr_array((1, 1), dtype=np.float64)`

## Reproducing the Bug

```python
import scipy.sparse
import scipy.io
import numpy as np

empty_sparse = scipy.sparse.csr_array((3, 3), dtype=np.float64)
scipy.io.hb_write("test_empty.hb", empty_sparse)
```

**Output:**
```
ValueError: zero-size array to reduction operation maximum which has no identity
```

**Full traceback:**
```
File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/_harwell_boeing/hb.py", line 561, in hb_write
    hb_info = HBInfo.from_data(m)
File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/_harwell_boeing/hb.py", line 79, in from_data
    indices_fmt = IntFormat.from_number(np.max(indices+1))
ValueError: zero-size array to reduction operation maximum which has no identity
```

## Why This Is A Bug

Empty sparse matrices are valid mathematical objects that can occur naturally when:
- Building sparse matrices incrementally
- Filtering out all non-zero elements
- Processing variable-sized sparse data

The documentation for `hb_write` does not exclude empty matrices, and the function should handle this edge case gracefully. The crash occurs because the code attempts to compute `np.max(indices+1)` when `indices` is an empty array, which raises a ValueError.

## Fix

The issue is in `scipy/io/_harwell_boeing/hb.py` at line 79 in the `HBInfo.from_data` method. The fix should handle the empty indices case:

```diff
--- a/scipy/io/_harwell_boeing/hb.py
+++ b/scipy/io/_harwell_boeing/hb.py
@@ -76,7 +76,10 @@ class HBInfo:
         # Compute integer format
         ptr_fmt = IntFormat.from_number(np.max(pointers))
-        indices_fmt = IntFormat.from_number(np.max(indices+1))
+        if len(indices) > 0:
+            indices_fmt = IntFormat.from_number(np.max(indices+1))
+        else:
+            indices_fmt = IntFormat.from_number(1)

         # Compute values format
         if values_dtype.kind == 'f':
```

The fix provides a sensible default format when there are no indices to examine. An alternative would be to use the matrix dimensions to determine the format, but since there are no actual indices to write, a minimal format suffices.