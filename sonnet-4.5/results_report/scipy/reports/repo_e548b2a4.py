import scipy.sparse as sp
import numpy as np

print("Testing scipy.sparse COO Matrix Transpose Canonical Format Bug")
print("=" * 60)

# Test case 1: Matrix with duplicates that get summed
data = [1, 2]
rows = [0, 0]
cols = [0, 0]
A = sp.coo_matrix((data, (rows, cols)), shape=(2, 2))

print("\nTest Case 1: Matrix with duplicate entries")
print("-" * 40)
print(f"Before sum_duplicates: A.has_canonical_format = {A.has_canonical_format}")

A.sum_duplicates()
print(f"After sum_duplicates: A.has_canonical_format = {A.has_canonical_format}")

A_T = A.transpose()
print(f"After transpose: A_T.has_canonical_format = {A_T.has_canonical_format}")

# Verify the transposed matrix actually has no duplicates
print(f"\nVerification:")
print(f"A_T.data = {A_T.data}")
print(f"A_T.row = {A_T.row}")
print(f"A_T.col = {A_T.col}")

# Test case 2: Matrix without duplicates from the start
print("\n" + "=" * 60)
print("\nTest Case 2: Matrix without duplicates")
print("-" * 40)

data2 = [1, 2, 3]
rows2 = [0, 1, 2]
cols2 = [0, 1, 2]
B = sp.coo_matrix((data2, (rows2, cols2)), shape=(3, 3))

print(f"Before sum_duplicates: B.has_canonical_format = {B.has_canonical_format}")
B.sum_duplicates()
print(f"After sum_duplicates: B.has_canonical_format = {B.has_canonical_format}")

B_T = B.transpose()
print(f"After transpose: B_T.has_canonical_format = {B_T.has_canonical_format}")

# Verify the transposed matrix structure
print(f"\nVerification:")
print(f"B_T.data = {B_T.data}")
print(f"B_T.row = {B_T.row}")
print(f"B_T.col = {B_T.col}")

# Show that calling sum_duplicates on transposed matrix changes only the flag
print("\n" + "=" * 60)
print("\nTest: Effect of sum_duplicates on already canonical transposed matrix")
print("-" * 40)

print(f"Before B_T.sum_duplicates(): B_T.has_canonical_format = {B_T.has_canonical_format}")
B_T_data_before = B_T.data.copy()
B_T_row_before = B_T.row.copy()
B_T_col_before = B_T.col.copy()

B_T.sum_duplicates()
print(f"After B_T.sum_duplicates(): B_T.has_canonical_format = {B_T.has_canonical_format}")

print(f"\nData changed: {not np.array_equal(B_T_data_before, B_T.data)}")
print(f"Row indices changed: {not np.array_equal(B_T_row_before, B_T.row)}")
print(f"Col indices changed: {not np.array_equal(B_T_col_before, B_T.col)}")