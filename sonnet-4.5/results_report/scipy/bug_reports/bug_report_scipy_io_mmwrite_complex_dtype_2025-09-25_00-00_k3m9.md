# Bug Report: scipy.io.mmwrite Complex Dtype Loss on Empty Sparse Matrices

**Target**: `scipy.io.mmwrite` / `scipy.io.mmread`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When writing a sparse complex matrix with all zeros (nnz=0) to a Matrix Market file using `mmwrite`, the field type is incorrectly written as "real" instead of "complex". This causes the dtype to be lost when reading the file back with `mmread`, violating the round-trip property.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_array
import tempfile
import os
import numpy as np

@given(
    rows=st.integers(min_value=1, max_value=100),
    cols=st.integers(min_value=1, max_value=100),
    density=st.floats(min_value=0.0, max_value=0.5),
)
@settings(max_examples=50)
def test_mmio_sparse_roundtrip_complex(rows, cols, density):
    np.random.seed(42)
    real_part = np.random.rand(rows, cols)
    imag_part = np.random.rand(rows, cols)
    data = real_part + 1j * imag_part
    mask = np.random.rand(rows, cols) < density
    data = data * mask
    sparse_matrix = coo_array(data)

    with tempfile.NamedTemporaryFile(suffix='.mtx', delete=False, mode='w') as f:
        filename = f.name

    try:
        mmwrite(filename, sparse_matrix)
        result = mmread(filename, spmatrix=False)

        assert isinstance(result, coo_array)
        assert result.shape == sparse_matrix.shape
        assert result.dtype == sparse_matrix.dtype

        np.testing.assert_allclose(result.toarray(), sparse_matrix.toarray(), rtol=1e-6)
    finally:
        if os.path.exists(filename):
            os.remove(filename)
```

**Failing input**: `rows=1, cols=1, density=0.0` (or any configuration that results in nnz=0)

## Reproducing the Bug

```python
import tempfile
import os
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_array

data = np.array([[0.0 + 0.0j]])
sparse_matrix = coo_array(data)

print(f"Original dtype: {sparse_matrix.dtype}")
print(f"Original nnz: {sparse_matrix.nnz}")

with tempfile.NamedTemporaryFile(suffix='.mtx', delete=False, mode='w') as f:
    filename = f.name

try:
    mmwrite(filename, sparse_matrix)

    with open(filename, 'r') as f:
        header = f.readline()
        print(f"File header: {header.strip()}")

    result = mmread(filename, spmatrix=False)

    print(f"Result dtype: {result.dtype}")
    print(f"Dtypes match: {result.dtype == sparse_matrix.dtype}")
finally:
    if os.path.exists(filename):
        os.remove(filename)
```

Output:
```
Original dtype: complex128
Original nnz: 0
File header: %%MatrixMarket matrix coordinate real symmetric
Result dtype: float64
Dtypes match: False
```

## Why This Is A Bug

1. The file header says "real" when the input matrix has a complex dtype
2. Reading back the file produces float64 instead of complex128
3. This violates the fundamental round-trip property: `mmread(mmwrite(matrix))` should preserve the dtype
4. The bug only occurs when the sparse matrix has all zeros (nnz=0)

## Fix

The issue is in the C++ implementation (`_fmm_core`) of scipy.io's fast_matrix_market backend. When writing a sparse matrix with nnz=0, the field type should be determined from the matrix's dtype (specifically `sparse_matrix.data.dtype`), not from the actual values in the matrix.

The field type determination logic needs to check the dtype even when the data array is empty:

```python
if field is None and len(data) == 0:
    if data.dtype.kind == 'c':
        field = 'complex'
    elif data.dtype.kind == 'f':
        field = 'real'
    elif data.dtype.kind == 'i':
        field = 'integer'
```

This logic should be added before the header is written to ensure that empty complex sparse matrices are correctly identified as "complex" in the Matrix Market file.