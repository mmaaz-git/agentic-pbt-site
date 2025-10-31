import tempfile
import numpy as np
from scipy.io import hb_write, hb_read
from scipy.sparse import coo_array
import traceback

# Test 1: Reproduce the basic bug
print("Test 1: Reproducing the basic bug with empty sparse matrix")
try:
    sparse_matrix = coo_array(([], ([], [])), shape=(5, 5))
    print(f"Created empty sparse matrix with shape: {sparse_matrix.shape}, nnz: {sparse_matrix.nnz}")

    with tempfile.NamedTemporaryFile(suffix='.hb', delete=False, mode='w') as f:
        filepath = f.name

    hb_write(filepath, sparse_matrix)
    print("SUCCESS: hb_write did not crash")
except Exception as e:
    print(f"CRASH: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test 2: Run the hypothesis test with the failing input
print("Test 2: Running hypothesis test with failing input (rows=1, cols=1, nnz=0)")

def test_hb_write_hb_read_roundtrip_manual(rows, cols, nnz):
    row_indices = np.random.randint(0, rows, size=nnz) if nnz > 0 else np.array([])
    col_indices = np.random.randint(0, cols, size=nnz) if nnz > 0 else np.array([])
    data = np.random.rand(nnz) if nnz > 0 else np.array([])

    sparse_matrix = coo_array((data, (row_indices, col_indices)), shape=(rows, cols))

    with tempfile.NamedTemporaryFile(suffix='.hb', delete=False, mode='w') as f:
        filepath = f.name

    try:
        hb_write(filepath, sparse_matrix)
        result = hb_read(filepath, spmatrix=False)

        assert result.shape == sparse_matrix.shape
        result_dense = result.toarray()
        expected_dense = sparse_matrix.toarray()
        assert np.allclose(result_dense, expected_dense)
        return True
    except Exception as e:
        print(f"Failed with rows={rows}, cols={cols}, nnz={nnz}: {e}")
        raise
    finally:
        import os
        if os.path.exists(filepath):
            os.unlink(filepath)

# Test the specific failing case
print("Testing with rows=1, cols=1, nnz=0:")
try:
    test_hb_write_hb_read_roundtrip_manual(1, 1, 0)
    print("SUCCESS: Test passed")
except Exception as e:
    print(f"FAILURE: Test failed with error")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test 3: Verify that non-empty matrices work fine
print("Test 3: Verify non-empty matrices work correctly")
try:
    sparse_matrix = coo_array(([1.0, 2.0], ([0, 1], [1, 2])), shape=(3, 3))
    print(f"Created sparse matrix with shape: {sparse_matrix.shape}, nnz: {sparse_matrix.nnz}")

    with tempfile.NamedTemporaryFile(suffix='.hb', delete=False, mode='w') as f:
        filepath = f.name

    hb_write(filepath, sparse_matrix)
    result = hb_read(filepath, spmatrix=False)
    print(f"Successfully wrote and read matrix")
    print(f"Original shape: {sparse_matrix.shape}, Read shape: {result.shape}")
    print(f"Data preserved correctly: {np.allclose(result.toarray(), sparse_matrix.toarray())}")

    import os
    os.unlink(filepath)
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()