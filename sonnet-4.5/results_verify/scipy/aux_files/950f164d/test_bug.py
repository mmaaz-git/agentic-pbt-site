import scipy.sparse as sp
import numpy as np

# Test the simple reproduction case
data = [1, 2]
rows = [0, 0]
cols = [0, 0]
A = sp.coo_matrix((data, (rows, cols)), shape=(2, 2))

print(f"Before sum_duplicates: A.has_canonical_format = {A.has_canonical_format}")

A.sum_duplicates()

print(f"After sum_duplicates: A.has_canonical_format = {A.has_canonical_format}")

A_T = A.transpose()
print(f"After transpose: A_T.has_canonical_format = {A_T.has_canonical_format}")

# Let's also test with a matrix that has no duplicates from the start
print("\nTesting with a matrix that has no duplicates:")
data2 = [1, 2, 3]
rows2 = [0, 1, 2]
cols2 = [0, 1, 2]
B = sp.coo_matrix((data2, (rows2, cols2)), shape=(3, 3))

print(f"Before sum_duplicates: B.has_canonical_format = {B.has_canonical_format}")
B.sum_duplicates()
print(f"After sum_duplicates: B.has_canonical_format = {B.has_canonical_format}")

B_T = B.transpose()
print(f"After transpose: B_T.has_canonical_format = {B_T.has_canonical_format}")