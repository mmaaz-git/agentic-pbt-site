# Bug Report: scipy.sparse.csr_matrix has_sorted_indices Flag Not Invalidated

**Target**: `scipy.sparse.csr_matrix.has_sorted_indices`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `has_sorted_indices` flag in `scipy.sparse.csr_matrix` (and `csc_matrix`) is not invalidated when the underlying indices array is modified directly, leading to a violated contract where the flag indicates sorted indices but the actual indices may not be sorted.

## Property-Based Test

```python
import numpy as np
import scipy.sparse as sp
from hypothesis import given, strategies as st, settings

@st.composite
def csr_matrices(draw):
    n = draw(st.integers(min_value=2, max_value=20))
    m = draw(st.integers(min_value=2, max_value=20))
    density = draw(st.floats(min_value=0.1, max_value=0.4))
    return sp.random(n, m, density=density, format='csr')

@given(csr_matrices())
@settings(max_examples=50)
def test_sorted_indices_flag_invalidation(A):
    A.sort_indices()
    assert A.has_sorted_indices

    if A.nnz >= 2:
        A.indices[0], A.indices[1] = A.indices[1], A.indices[0]

        indices_actually_sorted = all(
            np.all(A.indices[A.indptr[i]:A.indptr[i+1]][:-1] <=
                   A.indices[A.indptr[i]:A.indptr[i+1]][1:])
            for i in range(A.shape[0]) if A.indptr[i+1] - A.indptr[i] > 1
        )

        if A.has_sorted_indices and not indices_actually_sorted:
            raise AssertionError("BUG: has_sorted_indices flag not invalidated after indices modification")
```

**Failing input**: Any CSR matrix with at least 2 non-zero elements after calling `sort_indices()`

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp

data = np.array([1.0, 2.0, 3.0])
indices = np.array([2, 0, 1])
indptr = np.array([0, 3, 3])
A = sp.csr_matrix((data, indices, indptr), shape=(2, 3))

A.sort_indices()
print(f"After sort_indices: has_sorted_indices = {A.has_sorted_indices}")
print(f"Indices: {A.indices}")

A.indices[0], A.indices[1] = A.indices[1], A.indices[0]
print(f"\nAfter swapping indices[0] and indices[1]:")
print(f"has_sorted_indices = {A.has_sorted_indices}")
print(f"Indices: {A.indices}")
print(f"Actually sorted: {np.all(A.indices[:-1] <= A.indices[1:])}")
```

Output:
```
After sort_indices: has_sorted_indices = True
Indices: [0 1 2]

After swapping indices[0] and indices[1]:
has_sorted_indices = True
Indices: [1 0 2]
Actually sorted: False
```

## Why This Is A Bug

The `has_sorted_indices` attribute is documented to indicate whether the column indices within each row (for CSR) or row indices within each column (for CSC) are sorted. This flag is used for performance optimizations in various sparse matrix operations.

When `sort_indices()` is called, this flag is set to `True`. However, scipy.sparse allows direct modification of the underlying `indices` array. When this array is modified, the `has_sorted_indices` flag should be invalidated because:

1. The sorted indices guarantee is now violated
2. Algorithms that rely on sorted indices for correctness or performance will fail or produce incorrect results
3. Binary search operations in sparse matrix operations assume sorted indices
4. This violates the API contract that the flag accurately reflects the indices state

This is a **contract violation bug** that can lead to **silent correctness issues** in operations that depend on sorted indices.

## Fix

Similar to the canonical format bug, there are two possible fixes:

**Option 1: Make indices array read-only after sorting** (Recommended)

```diff
--- a/scipy/sparse/_compressed.py
+++ b/scipy/sparse/_compressed.py
@@ -1150,6 +1150,7 @@ class _cs_matrix(_data_matrix, _minmax_mixin):
             self.data[:] = self.data[perm]
             self.indices[:] = self.indices[perm]
             self.has_sorted_indices = True
+            self.indices.flags.writeable = False
```

**Option 2: Use property setters to invalidate the flag**

```diff
--- a/scipy/sparse/_compressed.py
+++ b/scipy/sparse/_compressed.py
@@ -50,7 +50,15 @@ class _cs_matrix(_data_matrix, _minmax_mixin):
     def __init__(self, arg1, shape=None, dtype=None, copy=False):
         # ... existing code ...
-        self.indices = indices
+        self._indices = indices
+
+    @property
+    def indices(self):
+        return self._indices
+
+    @indices.setter
+    def indices(self, value):
+        self._indices = value
+        self.has_sorted_indices = False
```

The first option prevents the bug by making the indices array immutable after sorting. The second option allows modifications but correctly invalidates the flag. Both fixes ensure the contract is maintained.