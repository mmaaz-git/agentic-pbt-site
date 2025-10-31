import scipy.sparse
import scipy.io
import numpy as np

# Create an empty sparse matrix
empty_sparse = scipy.sparse.csr_array((3, 3), dtype=np.float64)

print(f"Empty sparse matrix shape: {empty_sparse.shape}")
print(f"Empty sparse matrix nnz: {empty_sparse.nnz}")
print(f"Empty sparse matrix data: {empty_sparse.data}")
print(f"Empty sparse matrix indices: {empty_sparse.indices}")
print(f"Empty sparse matrix indptr: {empty_sparse.indptr}")
print()

scipy.io.hb_write("test_empty.hb", empty_sparse)