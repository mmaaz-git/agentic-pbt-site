# Bug Report: scipy.sparse COO Matrix Canonical Format Flag Lost After tocsr().tocoo()

**Target**: `scipy.sparse.coo_matrix`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When converting a COO matrix to CSR and back to COO, the resulting COO matrix loses its `has_canonical_format` flag even when it contains no duplicate coordinates. This causes the internal state flag to be incorrect, potentially leading to performance issues or incorrect behavior in code that relies on this flag.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy as np
from scipy import sparse

@given(
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False,
                           min_value=-100, max_value=100),
                  min_size=1, max_size=100),
    rows=st.lists(st.integers(min_value=0, max_value=99),
                  min_size=1, max_size=100),
    cols=st.lists(st.integers(min_value=0, max_value=99),
                  min_size=1, max_size=100)
)
def test_canonical_format_after_tocsr_tocoo(data, rows, cols):
    min_len = min(len(data), len(rows), len(cols))
    data = np.array(data[:min_len])
    rows = rows[:min_len]
    cols = cols[:min_len]

    assume(min_len > 0)

    shape = (100, 100)
    original = sparse.coo_matrix((data, (rows, cols)), shape=shape)
    converted = original.tocsr().tocoo()

    coords = list(zip(converted.row.tolist(), converted.col.tolist()))
    has_duplicates = len(coords) != len(set(coords))

    if not has_duplicates:
        assert converted.has_canonical_format, \
            "COO from CSR should have canonical format when no duplicates"
```

**Failing input**: Single element COO matrix: `data=[0.0], rows=[0], cols=[0]`

## Reproducing the Bug

```python
import numpy as np
from scipy import sparse

data = [1.0]
rows = [0]
cols = [0]
shape = (100, 100)

original_coo = sparse.coo_matrix((data, (rows, cols)), shape=shape)
csr = original_coo.tocsr()
coo_from_csr = csr.tocoo()

print(f"has_canonical_format: {coo_from_csr.has_canonical_format}")

coords = list(zip(coo_from_csr.row.tolist(), coo_from_csr.col.tolist()))
has_duplicates = len(coords) != len(set(coords))

print(f"Actually has duplicates: {has_duplicates}")
```

**Output:**
```
has_canonical_format: False
Actually has duplicates: False
```

**Expected:** `has_canonical_format: True` because the matrix has no duplicate coordinates.

## Why This Is A Bug

The `has_canonical_format` flag is part of the COO matrix's public API and should accurately reflect whether the matrix has duplicate coordinates. CSR matrices inherently cannot have duplicate coordinates due to their storage format, so any COO matrix created from CSR should have `has_canonical_format=True`.

This bug can cause:
1. Performance degradation - code may unnecessarily call `sum_duplicates()`
2. Incorrect assumptions - libraries relying on this flag for optimization
3. API contract violation - the flag doesn't match the actual state

## Fix

The bug is in the `tocoo()` method of CSR/CSC matrices. When creating a COO matrix from CSR/CSC format, the `has_canonical_format` flag should be set to `True` since CSR/CSC formats guarantee no duplicate coordinates.

```diff
--- a/scipy/sparse/_compressed.py
+++ b/scipy/sparse/_compressed.py
@@ -XXX,X +XXX,X @@ class _cs_matrix(_data_matrix, _minmax_mixin):
     def tocoo(self, copy=False):
         major_dim, minor_dim = self._swap(self.shape)
         minor_indices = self.indices.copy() if copy else self.indices
         major_indices = np.empty(len(minor_indices), dtype=self.indices.dtype)
         _sparsetools.expandptr(major_dim, self.indptr, major_indices)
         data = self.data.copy() if copy else self.data

         from ._coo import coo_matrix
         A = coo_matrix(
             (data, (major_indices, minor_indices)),
             self.shape, copy=False, dtype=self.dtype
         )
+        A.has_canonical_format = True
         A.has_sorted_indices = True
         return self._swap(A)
```

Note: The exact line numbers and method location may vary by scipy version. The key change is setting `has_canonical_format = True` when creating COO from CSR/CSC.