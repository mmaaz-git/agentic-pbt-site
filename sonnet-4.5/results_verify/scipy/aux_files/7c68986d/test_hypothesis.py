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
        assert result.dtype == sparse_matrix.dtype, f"Expected dtype {sparse_matrix.dtype}, got {result.dtype} (nnz={sparse_matrix.nnz})"

        np.testing.assert_allclose(result.toarray(), sparse_matrix.toarray(), rtol=1e-6)
    finally:
        if os.path.exists(filename):
            os.remove(filename)

# Test manually with specific failing example
print("Testing with specific failing input: rows=1, cols=1, density=0.0")

rows, cols, density = 1, 1, 0.0
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

    assert isinstance(result, coo_array), f"Expected coo_array, got {type(result)}"
    assert result.shape == sparse_matrix.shape, f"Shape mismatch: {result.shape} vs {sparse_matrix.shape}"

    if result.dtype != sparse_matrix.dtype:
        print(f"Test failed: Expected dtype {sparse_matrix.dtype}, got {result.dtype} (nnz={sparse_matrix.nnz})")
    else:
        np.testing.assert_allclose(result.toarray(), sparse_matrix.toarray(), rtol=1e-6)
        print("Test passed (no assertion error)")
finally:
    if os.path.exists(filename):
        os.remove(filename)

print("\nRunning hypothesis test...")
test_mmio_sparse_roundtrip_complex()