import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import csgraph_from_dense

test_values = [1e-300, 1e-100, 1e-50, 1e-20, 1e-10, 1e-5, 1e-3]

for val in test_values:
    graph = np.array([[0.0, val], [0.0, 0.0]])

    sparse_scipy = csr_array(graph)
    sparse_csgraph = csgraph_from_dense(graph, null_value=0)

    print(f"Value: {val:.2e}")
    print(f"  scipy.sparse.csr_array nnz: {sparse_scipy.nnz}")
    print(f"  csgraph_from_dense nnz: {sparse_csgraph.nnz}")
    print()