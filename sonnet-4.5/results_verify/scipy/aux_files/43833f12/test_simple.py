import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
import scipy.sparse as sp

graph = csr_matrix([[0, 1], [1, 0]], dtype=float)

lap = laplacian(graph, normed=False, form='array')

print(f"Type: {type(lap)}")
print(f"Is sparse: {sp.issparse(lap)}")

assert not sp.issparse(lap), f"Expected numpy array, got {type(lap).__name__}"