import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numpy as np

# Test spsolve with 1x1 matrices
A = sp.csr_matrix([[2.0]])
I = sp.eye(1, 1, format='csr')

print("Test spsolve with 1x1 sparse matrix RHS:")
print(f"A shape: {A.shape}, type: {type(A)}")
print(f"I shape: {I.shape}, type: {type(I)}")

result = spl.spsolve(A, I)
print(f"Result type: {type(result)}")
print(f"Result shape: {result.shape}")
print(f"Result value: {result}")
print(f"Is sparse: {sp.issparse(result)}")
print()

# Test with 2x2 matrices for comparison
A2 = sp.csr_matrix([[2.0, 0], [0, 3.0]])
I2 = sp.eye(2, 2, format='csr')

print("Test spsolve with 2x2 sparse matrix RHS:")
print(f"A2 shape: {A2.shape}, type: {type(A2)}")
print(f"I2 shape: {I2.shape}, type: {type(I2)}")

result2 = spl.spsolve(A2, I2)
print(f"Result type: {type(result2)}")
print(f"Result shape: {result2.shape}")
print(f"Is sparse: {sp.issparse(result2)}")
print()

# Test spsolve with 1x1 dense RHS
b = np.array([[1.0]])
print("Test spsolve with 1x1 dense matrix RHS:")
print(f"b shape: {b.shape}, type: {type(b)}")
result3 = spl.spsolve(A, b)
print(f"Result type: {type(result3)}")
print(f"Result shape: {result3.shape}")
print(f"Result value: {result3}")