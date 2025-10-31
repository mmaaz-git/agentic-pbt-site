import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

A_sparse = sp.csr_matrix([[1, 0], [0, 2]])
result = spl.expm(A_sparse)

print(f"Input type: {type(A_sparse)}")
print(f"Output type: {type(result)}")
print(f"Is output ndarray? {isinstance(result, np.ndarray)}")
print(f"Is output sparse? {sp.issparse(result)}")

# Also test with dense input
A_dense = np.array([[1, 0], [0, 2]])
result_dense = spl.expm(A_dense)
print(f"\nDense input type: {type(A_dense)}")
print(f"Dense output type: {type(result_dense)}")
print(f"Is dense output ndarray? {isinstance(result_dense, np.ndarray)}")