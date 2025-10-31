import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
import scipy.sparse as sp

# Test with dense array
dense_graph = np.array([[0, 1], [1, 0]], dtype=float)
lap_dense = laplacian(dense_graph, normed=False, form='array')
print(f"Dense input -> Type: {type(lap_dense)}, Is sparse: {sp.issparse(lap_dense)}")

# Test with sparse matrix
sparse_graph = csr_matrix([[0, 1], [1, 0]], dtype=float)
lap_sparse = laplacian(sparse_graph, normed=False, form='array')
print(f"Sparse input -> Type: {type(lap_sparse)}, Is sparse: {sp.issparse(lap_sparse)}")