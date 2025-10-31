import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Test what spsolve returns for sparse input
A = sp.csc_array([[2.0, 1.0], [1.0, 2.0]])
b = sp.csc_array([[1.0], [2.0]])

result = spsolve(A, b)
print(f"spsolve result type with sparse b: {type(result)}")
print(f"Result: {result}")

# With dense b
b_dense = np.array([1.0, 2.0])
result2 = spsolve(A, b_dense)
print(f"\nspsolve result type with dense b: {type(result2)}")
print(f"Result: {result2}")

# With matrix b (2D)
B = sp.csc_array([[1.0, 2.0], [2.0, 1.0]])
result3 = spsolve(A, B)
print(f"\nspsolve result type with sparse matrix B: {type(result3)}")
print(f"Result shape: {result3.shape}")
print(f"Is sparse: {sp.issparse(result3)}")
print(f"Result:\n{result3}")