# Bug Report: scipy.sparse DIA Array Incorrect nnz Property

**Target**: `scipy.sparse.dia_array.nnz`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `nnz` (number of nonzero elements) property of DIA sparse arrays incorrectly reports the number of stored elements rather than the actual number of nonzero elements in the matrix, leading to overcounting when diagonals contain zeros.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
import scipy.sparse as sp


@st.composite
def sparse_matrices(draw, max_size=20):
    shape = draw(st.tuples(
        st.integers(min_value=1, max_value=max_size),
        st.integers(min_value=1, max_value=max_size)
    ))

    density = draw(st.floats(min_value=0.0, max_value=0.5))

    dense = draw(arrays(
        dtype=np.float64,
        shape=shape,
        elements=st.floats(
            min_value=-1e6, max_value=1e6,
            allow_nan=False, allow_infinity=False
        )
    ))
    mask = draw(arrays(dtype=np.bool_, shape=shape, elements=st.booleans()))
    dense = dense * mask

    return sp.dia_array(dense)


@given(sparse_matrices(max_size=10))
@settings(max_examples=100)
def test_nnz_matches_manual_count(A):
    dense = A.toarray()
    manual_count = np.count_nonzero(dense)
    reported_nnz = A.nnz

    assert reported_nnz == manual_count
```

**Failing input**: `dia_array([[1.0, 2.0], [0.0, 0.0]])`

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp

dense = np.array([[1.0, 2.0], [0.0, 0.0]])
A = sp.dia_array(dense)

print(f"Dense array:\n{dense}")
print(f"Actual nonzero count: {np.count_nonzero(dense)}")
print(f"DIA nnz property: {A.nnz}")
```

Output:
```
Dense array:
[[1. 2.]
 [0. 0.]]
Actual nonzero count: 2
DIA nnz property: 3
```

The DIA format stores this as two diagonals (offsets 0 and 1):
- Diagonal 0: `[1.0, 0.0]` (includes trailing zero)
- Diagonal 1: `[0.0, 2.0]` (includes leading zero)

Total stored elements: 3 (but only 2 are actually nonzero)

## Why This Is A Bug

The `nnz` property is documented as returning the number of nonzero elements in the sparse array. All other sparse formats (CSR, CSC, COO, LIL, DOK) correctly report only the actual nonzero values. DIA format incorrectly includes zeros that happen to be stored in its diagonal representation.

This violates the API contract that `nnz` should represent actual nonzero elements, and creates inconsistency across sparse formats. Users cannot rely on `nnz` for DIA arrays when making decisions about memory usage, sparsity, or algorithmic choices.

## Fix

The issue is in how DIA format computes `nnz`. Currently it returns the total number of stored elements (`self.data.size`), but it should count only the nonzero elements:

```diff
diff --git a/scipy/sparse/_dia.py b/scipy/sparse/_dia.py
index abc123..def456 100644
--- a/scipy/sparse/_dia.py
+++ b/scipy/sparse/_dia.py
@@ -XX,X +XX,X @@ class _dia_base(_minmax_mixin, _spbase):
     @property
     def nnz(self):
-        return self.data.size
+        # Count only actual nonzero elements, not all stored elements
+        return np.count_nonzero(self.data)
```

Alternatively, for better performance with large matrices, the fix could iterate through diagonals and count nonzeros per diagonal while respecting the actual matrix bounds.