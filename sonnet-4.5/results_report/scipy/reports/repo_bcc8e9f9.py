import numpy as np
from scipy import sparse
from scipy.sparse import linalg

# Create sparse diagonal matrices of different sizes
A1 = sparse.diags([2.0], offsets=0, format='csr')
A2 = sparse.diags([2.0, 3.0], offsets=0, format='csr')
A3 = sparse.diags([1.0, 2.0, 3.0], offsets=0, format='csr')

# Compute the inverse of each matrix
inv1 = linalg.inv(A1)
inv2 = linalg.inv(A2)
inv3 = linalg.inv(A3)

# Print the types
print("1x1 matrix inverse type:", type(inv1))
print("2x2 matrix inverse type:", type(inv2))
print("3x3 matrix inverse type:", type(inv3))

# Check if they are sparse
print("\n1x1 is sparse:", sparse.issparse(inv1))
print("2x2 is sparse:", sparse.issparse(inv2))
print("3x3 is sparse:", sparse.issparse(inv3))

# Print the actual values
print("\n1x1 inverse value:", inv1)
print("2x2 inverse diagonal:", inv2.diagonal())
print("3x3 inverse diagonal:", inv3.diagonal())

# Demonstrate the error this can cause
print("\nTrying to call .toarray() on each result:")
try:
    print("1x1 toarray:", inv1.toarray())
except AttributeError as e:
    print("1x1 toarray failed:", e)

try:
    print("2x2 toarray shape:", inv2.toarray().shape)
except AttributeError as e:
    print("2x2 toarray failed:", e)

try:
    print("3x3 toarray shape:", inv3.toarray().shape)
except AttributeError as e:
    print("3x3 toarray failed:", e)