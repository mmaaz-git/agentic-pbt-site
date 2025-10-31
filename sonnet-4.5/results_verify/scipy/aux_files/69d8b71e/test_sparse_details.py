import numpy as np
from scipy.sparse import csc_array

# Create an all-zero sparse matrix (5x5)
A = csc_array(np.zeros((5, 5)))

print(f"Shape: {A.shape}")
print(f"Number of non-zeros (nnz): {A.nnz}")
print(f"Data array: {A.data}")
print(f"Data array length: {len(A.data)}")
print(f"Indices array: {A.indices}")
print(f"Indices array length: {len(A.indices)}")
print(f"Indptr array: {A.indptr}")
print(f"Indptr array length: {len(A.indptr)}")

# What happens with np.max on empty array?
print("\nTesting np.max on empty array:")
try:
    result = np.max(A.indices)
    print(f"np.max(A.indices) = {result}")
except ValueError as e:
    print(f"ValueError: {e}")

try:
    result = np.max(A.indices + 1)
    print(f"np.max(A.indices + 1) = {result}")
except ValueError as e:
    print(f"ValueError: {e}")