# Bug Report: scipy.io.mmwrite Complex Dtype Loss for Empty Sparse Matrices

**Target**: `scipy.io.mmwrite` / `scipy.io.mmread`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When writing a complex sparse matrix with zero non-zero elements (nnz=0) to a Matrix Market file using `mmwrite`, the file header incorrectly specifies "real" instead of "complex" as the data type. This causes the matrix to be read back as float64 instead of complex128, resulting in silent data type loss.

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
    """Test that scipy.io.mmwrite/mmread preserves complex dtype for sparse matrices"""
    np.random.seed(42)

    # Create complex sparse matrix
    real_part = np.random.rand(rows, cols)
    imag_part = np.random.rand(rows, cols)
    data = real_part + 1j * imag_part

    # Apply sparsity mask
    mask = np.random.rand(rows, cols) < density
    data = data * mask
    sparse_matrix = coo_array(data)

    # Write to and read from Matrix Market file
    with tempfile.NamedTemporaryFile(suffix='.mtx', delete=False, mode='w') as f:
        filename = f.name

    try:
        mmwrite(filename, sparse_matrix)
        result = mmread(filename, spmatrix=False)

        # Check that properties are preserved
        assert isinstance(result, coo_array), f"Expected coo_array, got {type(result)}"
        assert result.shape == sparse_matrix.shape, f"Shape mismatch: {result.shape} != {sparse_matrix.shape}"
        assert result.dtype == sparse_matrix.dtype, f"Dtype mismatch: {result.dtype} != {sparse_matrix.dtype} (nnz={sparse_matrix.nnz})"

        # Check that values match
        np.testing.assert_allclose(result.toarray(), sparse_matrix.toarray(), rtol=1e-6)

    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    # Run the property-based test
    print("Running property-based test for scipy.io.mmwrite/mmread with complex sparse matrices...")
    print("Testing multiple scenarios with random sizes and densities...\n")

    try:
        test_mmio_sparse_roundtrip_complex()
        print("\n✓ All tests passed!")
    except AssertionError as e:
        print(f"\n✗ Test failed with assertion error: {e}")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
```

<details>

<summary>
**Failing input**: `rows=1, cols=1, density=0.0`
</summary>
```
Running property-based test for scipy.io.mmwrite/mmread with complex sparse matrices...
Testing multiple scenarios with random sizes and densities...


Falsifying example found!
  rows=1, cols=1, density=0.0
  Original dtype: complex128
  Result dtype: float64
  nnz: 0
  Matrix Market header: %%MatrixMarket matrix coordinate real symmetric

✗ Test failed: Dtype mismatch: float64 != complex128 (nnz=0)
```
</details>

## Reproducing the Bug

```python
import tempfile
import os
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_array

# Create a complex sparse matrix with all zeros (nnz=0)
data = np.array([[0.0 + 0.0j]])
sparse_matrix = coo_array(data)

print(f"Original dtype: {sparse_matrix.dtype}")
print(f"Original nnz: {sparse_matrix.nnz}")
print(f"Original shape: {sparse_matrix.shape}")
print()

with tempfile.NamedTemporaryFile(suffix='.mtx', delete=False, mode='w') as f:
    filename = f.name

try:
    mmwrite(filename, sparse_matrix)

    with open(filename, 'r') as f:
        header = f.readline()
        print(f"File header written: {header.strip()}")
    print()

    result = mmread(filename, spmatrix=False)

    print(f"Result dtype: {result.dtype}")
    print(f"Result nnz: {result.nnz}")
    print(f"Result shape: {result.shape}")
    print()
    print(f"Dtypes match: {result.dtype == sparse_matrix.dtype}")
    print(f"Expected dtype: complex128, Got dtype: {result.dtype}")

    # Additional test with non-zero complex matrix
    print("\n--- Testing with non-zero complex matrix ---")
    data_nonzero = np.array([[1.0 + 2.0j]])
    sparse_matrix_nonzero = coo_array(data_nonzero)

    print(f"Non-zero original dtype: {sparse_matrix_nonzero.dtype}")
    print(f"Non-zero original nnz: {sparse_matrix_nonzero.nnz}")

    with tempfile.NamedTemporaryFile(suffix='.mtx', delete=False, mode='w') as f2:
        filename2 = f2.name

    mmwrite(filename2, sparse_matrix_nonzero)

    with open(filename2, 'r') as f:
        header2 = f.readline()
        print(f"Non-zero file header: {header2.strip()}")

    result_nonzero = mmread(filename2, spmatrix=False)
    print(f"Non-zero result dtype: {result_nonzero.dtype}")
    print(f"Non-zero dtypes match: {result_nonzero.dtype == sparse_matrix_nonzero.dtype}")

    os.remove(filename2)

finally:
    if os.path.exists(filename):
        os.remove(filename)
```

<details>

<summary>
Output demonstrating the dtype loss for empty complex matrices
</summary>
```
Original dtype: complex128
Original nnz: 0
Original shape: (1, 1)

File header written: %%MatrixMarket matrix coordinate real symmetric

Result dtype: float64
Result nnz: 0
Result shape: (1, 1)

Dtypes match: False
Expected dtype: complex128, Got dtype: float64

--- Testing with non-zero complex matrix ---
Non-zero original dtype: complex128
Non-zero original nnz: 1
Non-zero file header: %%MatrixMarket matrix coordinate complex symmetric
Non-zero result dtype: complex128
Non-zero dtypes match: True
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Matrix Market Format Violation**: The Matrix Market format specification (NIST) explicitly requires the data type (real, complex, integer, pattern) to be correctly specified in the header line. Writing "real" for a complex matrix violates the format specification.

2. **Round-trip Property Violation**: A fundamental expectation of any serialization format is that `deserialize(serialize(data))` should preserve the data's properties. Here, `mmread(mmwrite(matrix))` fails to preserve the dtype for empty complex sparse matrices.

3. **Inconsistent Behavior**: The same complex sparse matrix writes "complex" when it has non-zero values but "real" when empty. This inconsistency indicates a logic error rather than intentional design.

4. **Silent Data Loss**: The dtype change from complex128 to float64 happens silently without any warning, potentially causing type errors in downstream computations that expect complex numbers.

5. **Documentation Expectation**: While scipy's documentation doesn't explicitly guarantee dtype preservation, the function already correctly handles complex matrices with non-zero values, suggesting this capability is intended.

## Relevant Context

- The bug occurs in scipy's fast_matrix_market implementation (`scipy.io._fast_matrix_market`), which is the default since scipy 1.12.0.
- The issue is in the C++ backend (`_fmm_core`) which determines the field type from the actual data values rather than the dtype when the data array is empty.
- This only affects sparse matrices (coo_array/coo_matrix) with complex dtype and exactly zero non-zero elements.
- Dense arrays and real-valued sparse matrices are not affected.
- Workaround available: Explicitly specify `field='complex'` when calling mmwrite.

Documentation references:
- Matrix Market format specification: http://math.nist.gov/MatrixMarket/formats.html
- scipy.io.mmwrite: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmwrite.html

## Proposed Fix

The fix requires modifying the C++ implementation in `_fmm_core` to check the dtype of the sparse matrix's data array even when it's empty. When `field` is not explicitly provided and the data array has zero elements, the field type should be determined from `sparse_matrix.data.dtype.kind`:

```diff
# Conceptual fix in the C++ backend (pseudo-code)
if field is None:
    if data.size() == 0:
+       # Check dtype even for empty arrays
+       if data.dtype().kind() == 'c':
+           field = "complex"
+       elif data.dtype().kind() == 'f':
+           field = "real"
+       elif data.dtype().kind() == 'i':
+           field = "integer"
+       else:
+           field = "real"  # default
    else:
        # Existing logic to infer from data values
        field = infer_field_from_values(data)
```

Alternatively, a Python-level fix in `scipy/io/_fast_matrix_market/__init__.py`:

```diff
def mmwrite(target, a, comment=None, field=None, precision=None, symmetry="AUTO"):
    # ... existing code ...
    elif issparse(a):
        # Write sparse scipy matrices
        a = a.tocoo()

+       # Determine field from dtype if not specified and data is empty
+       if field is None and a.nnz == 0:
+           if a.data.dtype.kind == 'c':
+               field = 'complex'
+           elif a.data.dtype.kind == 'f':
+               field = 'real'
+           elif a.data.dtype.kind == 'i':
+               field = 'integer'

        if symmetry is not None and symmetry != "general":
            # ... existing code ...
```