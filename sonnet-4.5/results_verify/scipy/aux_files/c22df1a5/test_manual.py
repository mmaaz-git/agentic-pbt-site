import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

A_sparse = sp.csc_array([[1.0, 0.0], [0.0, 2.0]])
result = sla.expm(A_sparse)

print(f"Input type: {type(A_sparse).__name__}")
print(f"Output type: {type(result).__name__}")
print(f"Documentation says: 'Returns expA : (M,M) ndarray'")
print(f"Is result an ndarray? {isinstance(result, np.ndarray)}")
print(f"Is result a sparse array? {sp.issparse(result)}")

# Test with dense input
A_dense = np.array([[1.0, 0.0], [0.0, 2.0]])
result_dense = sla.expm(A_dense)
print(f"\nDense input type: {type(A_dense).__name__}")
print(f"Dense output type: {type(result_dense).__name__}")
print(f"Is dense result an ndarray? {isinstance(result_dense, np.ndarray)}")