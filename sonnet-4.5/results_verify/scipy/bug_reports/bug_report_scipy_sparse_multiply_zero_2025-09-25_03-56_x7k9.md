# Bug Report: scipy.sparse Scalar Multiplication by Zero

**Target**: `scipy.sparse` scalar multiplication
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Multiplying a sparse matrix by zero leaves explicit zero values in the sparse structure, wasting memory and violating the sparse matrix principle of only storing nonzero values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.sparse as sp

@st.composite
def sparse_matrices(draw):
    rows = draw(st.integers(min_value=1, max_value=20))
    cols = draw(st.integers(min_value=1, max_value=20))
    density = draw(st.floats(min_value=0.0, max_value=1.0))
    dense_array = np.random.rand(rows, cols)
    dense_array[dense_array > density] = 0
    return sp.csr_matrix(dense_array)

@given(sparse_matrices())
@settings(max_examples=300)
def test_multiply_zero_should_eliminate_stored_values(matrix):
    if matrix.nnz == 0:
        return

    result = matrix * 0
    assert result.nnz == 0, f"Expected nnz=0, got nnz={result.nnz}"
    assert np.all(result.toarray() == 0)
```

**Failing input**: Any sparse matrix with at least one stored value, e.g., `csr_matrix([[1.0]])`

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp

matrix = sp.csr_matrix([[5.0, 0.0], [0.0, 3.0]])
print(f"Original nnz: {matrix.nnz}")
print(f"Original data: {matrix.data}")

result = matrix * 0
print(f"After multiply by 0:")
print(f"  nnz: {result.nnz}")
print(f"  data: {result.data}")
print(f"  All values zero? {np.all(result.toarray() == 0)}")

print(f"Expected: nnz should be 0")
print(f"Actual: nnz is {result.nnz} with stored zeros: {result.data}")
```

**Output:**
```
Original nnz: 2
Original data: [5. 3.]
After multiply by 0:
  nnz: 2
  data: [0. 0.]
  All values zero? True
Expected: nnz should be 0
Actual: nnz is 2 with stored zeros: [0. 0.]
```

## Why This Is A Bug

1. **Violates sparse matrix efficiency principle**: Sparse matrices should only store nonzero values to save memory. Storing explicit zeros defeats this purpose.

2. **Misleading `nnz` property**: The property name `nnz` conventionally means "number of nonzeros", but after `matrix * 0`, the matrix has no actual nonzeros yet `nnz > 0`.

3. **Inconsistent with user expectations**: When users multiply by zero, they expect the result to be truly sparse (no stored values), not a matrix with explicit zeros.

4. **Memory waste**: For large sparse matrices, this wastes significant memory storing zeros that don't need to be stored.

5. **Scipy provides `eliminate_zeros()`**: The existence of this method indicates that explicit zeros are undesirable and should be removed. Scalar multiplication by zero should do this automatically.

6. **Performance impact**: Operations that iterate over stored values (like subsequent multiplications) will be slower due to processing stored zeros.

## Fix

The scalar multiplication operation should automatically eliminate zeros when the scalar is zero. Here's the conceptual fix:

```diff
diff --git a/scipy/sparse/_base.py b/scipy/sparse/_base.py
index xxxxx..yyyyy 100644
--- a/scipy/sparse/_base.py
+++ b/scipy/sparse/_base.py
@@ -400,6 +400,10 @@ class sparray:
     def __mul__(self, other):
         if np.isscalar(other):
             result = self._mul_scalar(other)
+            # Eliminate explicit zeros when multiplying by zero
+            if other == 0:
+                result.eliminate_zeros()
             return result
         elif issparse(other):
             ...
```

Alternatively, the fix could be in the `_mul_scalar` implementation for each sparse format (CSR, CSC, COO, etc.) to avoid creating the explicit zeros in the first place when the scalar is zero.