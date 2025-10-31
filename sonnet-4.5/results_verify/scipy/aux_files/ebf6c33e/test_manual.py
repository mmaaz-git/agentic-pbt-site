import numpy as np
from scipy import sparse

data = [1.0, 2.0]
rows = [0, 0]
cols = [0, 0]
shape = (100, 100)

mat = sparse.coo_matrix((data, (rows, cols)), shape=shape)
print(f"Before sum_duplicates: {mat.has_canonical_format}")

mat.sum_duplicates()
print(f"After sum_duplicates: {mat.has_canonical_format}")

copied = mat.copy()
print(f"After copy(): {copied.has_canonical_format}")

print(f"\nData identical: {np.allclose(mat.toarray(), copied.toarray())}")

# Also check the actual data
print(f"\nOriginal matrix data: {mat.data}")
print(f"Copied matrix data: {copied.data}")
print(f"Original matrix row indices: {mat.row}")
print(f"Copied matrix row indices: {copied.row}")
print(f"Original matrix col indices: {mat.col}")
print(f"Copied matrix col indices: {copied.col}")