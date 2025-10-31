import numpy as np
from scipy import sparse

# Create a COO matrix with duplicate coordinates
data = [1.0, 2.0]
rows = [0, 0]  # duplicate row index
cols = [0, 0]  # duplicate column index
shape = (100, 100)

# Create COO matrix
mat = sparse.coo_matrix((data, (rows, cols)), shape=shape)
print(f"Before sum_duplicates: has_canonical_format = {mat.has_canonical_format}")
print(f"Data shape before: {mat.data.shape}")

# Call sum_duplicates to merge duplicate entries
mat.sum_duplicates()
print(f"\nAfter sum_duplicates: has_canonical_format = {mat.has_canonical_format}")
print(f"Data shape after: {mat.data.shape}")
print(f"Data value: {mat.data}")

# Copy the matrix
copied = mat.copy()
print(f"\nAfter copy(): has_canonical_format = {copied.has_canonical_format}")
print(f"Copied data shape: {copied.data.shape}")
print(f"Copied data value: {copied.data}")

# Check if data is identical
print(f"\nData arrays equal: {np.array_equal(mat.data, copied.data)}")
print(f"Row indices equal: {np.array_equal(mat.row, copied.row)}")
print(f"Col indices equal: {np.array_equal(mat.col, copied.col)}")
print(f"Dense arrays equal: {np.allclose(mat.toarray(), copied.toarray())}")

# Bug demonstration: the canonical format flag is lost
print(f"\nBUG: Original has_canonical_format={mat.has_canonical_format}, Copy has_canonical_format={copied.has_canonical_format}")
print("Expected: Both should be True after copy()")