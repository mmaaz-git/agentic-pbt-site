import numpy as np
import scipy.sparse as sp

print("Testing the has_sorted_indices flag bug...")
print("=" * 50)

data = np.array([1.0, 2.0, 3.0])
indices = np.array([2, 0, 1])
indptr = np.array([0, 3, 3])
A = sp.csr_matrix((data, indices, indptr), shape=(2, 3))

print("Initial state:")
print(f"has_sorted_indices = {A.has_sorted_indices}")
print(f"Indices: {A.indices}")

A.sort_indices()
print(f"\nAfter sort_indices:")
print(f"has_sorted_indices = {A.has_sorted_indices}")
print(f"Indices: {A.indices}")

# Now swap indices
A.indices[0], A.indices[1] = A.indices[1], A.indices[0]
print(f"\nAfter swapping indices[0] and indices[1]:")
print(f"has_sorted_indices = {A.has_sorted_indices}")
print(f"Indices: {A.indices}")
print(f"Actually sorted: {np.all(A.indices[:-1] <= A.indices[1:])}")

# Check if this is a bug
if A.has_sorted_indices and not np.all(A.indices[:-1] <= A.indices[1:]):
    print("\n*** BUG CONFIRMED: has_sorted_indices is True but indices are not sorted ***")
else:
    print("\nNo bug found")