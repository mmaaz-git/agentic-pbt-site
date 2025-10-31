import numpy as np
import scipy.sparse as sp

# Create a CSR matrix with unsorted indices
data = np.array([1.0, 2.0, 3.0])
indices = np.array([2, 0, 1])
indptr = np.array([0, 3, 3])
A = sp.csr_matrix((data, indices, indptr), shape=(2, 3))

# Sort the indices
A.sort_indices()
print(f"After sort_indices():")
print(f"  has_sorted_indices = {A.has_sorted_indices}")
print(f"  Indices: {A.indices}")

# Verify indices are actually sorted
indices_sorted = np.all(A.indices[:-1] <= A.indices[1:])
print(f"  Indices actually sorted: {indices_sorted}")

# Now swap the first two indices to make them unsorted
print(f"\nSwapping indices[0] and indices[1]...")
A.indices[0], A.indices[1] = A.indices[1], A.indices[0]

# Check the state after modification
print(f"\nAfter direct modification:")
print(f"  has_sorted_indices = {A.has_sorted_indices}")
print(f"  Indices: {A.indices}")

# Verify indices are actually sorted (they shouldn't be)
indices_sorted = np.all(A.indices[:-1] <= A.indices[1:])
print(f"  Indices actually sorted: {indices_sorted}")

# This demonstrates the bug
if A.has_sorted_indices and not indices_sorted:
    print(f"\nBUG DETECTED: has_sorted_indices is {A.has_sorted_indices} but indices are NOT sorted!")
    print("This violates the contract that has_sorted_indices accurately reflects the state.")