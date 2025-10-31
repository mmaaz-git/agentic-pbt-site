import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

# Test with different input types
print("=== Testing different input types ===\n")

# Dense input
dense_A = np.array([[1.0, 0.0], [0.0, 2.0]])
result_dense = spl.expm(dense_A)
print(f"Dense input type: {type(dense_A)}")
print(f"Dense result type: {type(result_dense)}")
print(f"Is result an ndarray?: {isinstance(result_dense, np.ndarray)}")
print()

# Sparse CSR input
sparse_csr = sp.csr_array([[1.0, 0.0], [0.0, 2.0]])
result_csr = spl.expm(sparse_csr)
print(f"Sparse CSR input type: {type(sparse_csr)}")
print(f"Sparse CSR result type: {type(result_csr)}")
print(f"Is result an ndarray?: {isinstance(result_csr, np.ndarray)}")
print(f"Is result sparse?: {sp.issparse(result_csr)}")
print()

# Sparse CSC input
sparse_csc = sp.csc_array([[1.0, 0.0], [0.0, 2.0]])
result_csc = spl.expm(sparse_csc)
print(f"Sparse CSC input type: {type(sparse_csc)}")
print(f"Sparse CSC result type: {type(result_csc)}")
print(f"Is result an ndarray?: {isinstance(result_csc, np.ndarray)}")
print(f"Is result sparse?: {sp.issparse(result_csc)}")
print()

# Old matrix format (deprecated but still supported)
sparse_matrix = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
result_matrix = spl.expm(sparse_matrix)
print(f"Sparse matrix input type: {type(sparse_matrix)}")
print(f"Sparse matrix result type: {type(result_matrix)}")
print(f"Is result an ndarray?: {isinstance(result_matrix, np.ndarray)}")
print(f"Is result sparse?: {sp.issparse(result_matrix)}")