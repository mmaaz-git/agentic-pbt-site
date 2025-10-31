import scipy.sparse as sp
import numpy as np

# Create a matrix with duplicates, sum them, then transpose
data = [1, 2, 3]
rows = [0, 0, 1]
cols = [0, 0, 1]
A = sp.coo_matrix((data, (rows, cols)), shape=(3, 3))

print("Original matrix with duplicates:")
print(f"Data: {A.data}")
print(f"Row: {A.row}")
print(f"Col: {A.col}")
print(f"has_canonical_format: {A.has_canonical_format}")

A.sum_duplicates()
print("\nAfter sum_duplicates:")
print(f"Data: {A.data}")
print(f"Row: {A.row}")
print(f"Col: {A.col}")
print(f"has_canonical_format: {A.has_canonical_format}")

A_T = A.transpose()
print("\nAfter transpose:")
print(f"Data: {A_T.data}")
print(f"Row: {A_T.row}")
print(f"Col: {A_T.col}")
print(f"has_canonical_format: {A_T.has_canonical_format}")

# Check if there are actually any duplicates in the transposed matrix
coords = list(zip(A_T.row, A_T.col))
unique_coords = set(coords)
has_duplicates = len(coords) != len(unique_coords)
print(f"\nDoes transposed matrix have duplicate coordinates? {has_duplicates}")

# Let's also check if it's sorted
is_sorted = all(coords[i] <= coords[i+1] for i in range(len(coords)-1))
print(f"Is transposed matrix sorted by (row, col)? {is_sorted}")

# What if we call sum_duplicates on the transposed matrix?
A_T.sum_duplicates()
print(f"\nAfter calling sum_duplicates on transposed matrix:")
print(f"Data: {A_T.data}")
print(f"Row: {A_T.row}")
print(f"Col: {A_T.col}")
print(f"has_canonical_format: {A_T.has_canonical_format}")

# Did the data change at all?
print("\nDid sum_duplicates change anything? No - because there were no duplicates!")