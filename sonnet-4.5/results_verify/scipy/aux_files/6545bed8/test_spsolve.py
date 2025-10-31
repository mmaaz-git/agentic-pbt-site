import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

print("Testing spsolve behavior with different matrix sizes")
print("=" * 60)

# Test 1x1 matrix
A1 = sparse.diags([2.0], offsets=0, format='csr')
I1 = sparse.eye(1, format='csr')
result1 = spsolve(A1, I1)
print(f"1x1 case:")
print(f"  A shape: {A1.shape}, I shape: {I1.shape}")
print(f"  Result type: {type(result1)}")
print(f"  Result is sparse: {sparse.issparse(result1)}")
print(f"  Result value: {result1}")

print("\n" + "-" * 60 + "\n")

# Test 2x2 matrix
A2 = sparse.diags([2.0, 3.0], offsets=0, format='csr')
I2 = sparse.eye(2, format='csr')
result2 = spsolve(A2, I2)
print(f"2x2 case:")
print(f"  A shape: {A2.shape}, I shape: {I2.shape}")
print(f"  Result type: {type(result2)}")
print(f"  Result is sparse: {sparse.issparse(result2)}")
if sparse.issparse(result2):
    print(f"  Result value:\n{result2.toarray()}")
else:
    print(f"  Result value:\n{result2}")

print("\n" + "-" * 60 + "\n")

# Test 1x1 matrix with vector RHS
b1 = np.array([1.0])
result1_vec = spsolve(A1, b1)
print(f"1x1 case with vector RHS:")
print(f"  A shape: {A1.shape}, b shape: {b1.shape}")
print(f"  Result type: {type(result1_vec)}")
print(f"  Result value: {result1_vec}")

# Test if spsolve documentation mentions this behavior
print("\n" + "=" * 60)
print("\nChecking spsolve documentation:")
from scipy.sparse.linalg import spsolve
help(spsolve)