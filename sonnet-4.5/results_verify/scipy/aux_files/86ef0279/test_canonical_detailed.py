import numpy as np
import scipy.sparse as sp

print("=" * 60)
print("Testing what canonical format means")
print("=" * 60)

# Test 1: Create matrix with duplicates
data = np.array([1.0, 2.0, 3.0])
row = np.array([0, 0, 0])
col = np.array([0, 0, 1])
A = sp.coo_matrix((data, (row, col)), shape=(2, 2))

print(f"Matrix with duplicates at (0,0):")
print(f"  data: {A.data}, row: {A.row}, col: {A.col}")
print(f"  has_canonical_format: {A.has_canonical_format}")
print(f"  Dense representation:\n{A.todense()}")

# After sum_duplicates
A.sum_duplicates()
print(f"\nAfter sum_duplicates:")
print(f"  data: {A.data}, row: {A.row}, col: {A.col}")
print(f"  has_canonical_format: {A.has_canonical_format}")
print(f"  Dense representation:\n{A.todense()}")

# Test 2: Create matrix with unsorted indices
print("\n" + "=" * 60)
print("Testing matrix with unsorted indices")
print("=" * 60)

data2 = np.array([1.0, 2.0, 3.0])
row2 = np.array([1, 0, 0])  # Not sorted
col2 = np.array([0, 1, 0])  # Not sorted
B = sp.coo_matrix((data2, (row2, col2)), shape=(2, 2))

print(f"Matrix with unsorted indices:")
print(f"  data: {B.data}, row: {B.row}, col: {B.col}")
print(f"  has_canonical_format: {B.has_canonical_format}")

B.sum_duplicates()
print(f"\nAfter sum_duplicates (should sort and remove duplicates):")
print(f"  data: {B.data}, row: {B.row}, col: {B.col}")
print(f"  has_canonical_format: {B.has_canonical_format}")

# Test 3: Verify that modifying breaks canonical property
print("\n" + "=" * 60)
print("Testing if modification really breaks canonical format")
print("=" * 60)

data3 = np.array([1.0, 2.0, 3.0])
row3 = np.array([0, 1, 1])
col3 = np.array([0, 0, 1])
C = sp.coo_matrix((data3, (row3, col3)), shape=(2, 2))

C.sum_duplicates()
print(f"After sum_duplicates:")
print(f"  data: {C.data}, row: {C.row}, col: {C.col}")
print(f"  has_canonical_format: {C.has_canonical_format}")

# Modify to create duplicate
if len(C.row) > 1:
    C.row[1] = C.row[0]  # Make duplicate row index
    C.col[1] = C.col[0]  # Make duplicate col index
    print(f"\nAfter creating duplicate at ({C.row[0]},{C.col[0]}):")
    print(f"  data: {C.data}, row: {C.row}, col: {C.col}")
    print(f"  has_canonical_format: {C.has_canonical_format}")
    print(f"  This matrix now has duplicates but flag still says: {C.has_canonical_format}")

    # Verify it really has duplicates now
    print(f"\n  Converting to dense (will sum duplicates):\n{C.todense()}")

    # Create a fresh matrix with the same data to compare
    D = sp.coo_matrix((C.data, (C.row, C.col)), shape=C.shape)
    print(f"  Fresh matrix with same data has_canonical_format: {D.has_canonical_format}")

# Test 4: Check if modification breaks sorting
print("\n" + "=" * 60)
print("Testing if modification breaks sorting property")
print("=" * 60)

data4 = np.array([1.0, 2.0, 3.0])
row4 = np.array([0, 1, 1])
col4 = np.array([0, 0, 1])
E = sp.coo_matrix((data4, (row4, col4)), shape=(3, 3))

E.sum_duplicates()
print(f"After sum_duplicates (should be sorted):")
print(f"  data: {E.data}, row: {E.row}, col: {E.col}")
print(f"  has_canonical_format: {E.has_canonical_format}")

# Make it unsorted
if len(E.row) > 1:
    E.row[0], E.row[-1] = E.row[-1], E.row[0]
    E.col[0], E.col[-1] = E.col[-1], E.col[0]
    E.data[0], E.data[-1] = E.data[-1], E.data[0]
    print(f"\nAfter swapping first and last entries:")
    print(f"  data: {E.data}, row: {E.row}, col: {E.col}")
    print(f"  has_canonical_format: {E.has_canonical_format}")
    print(f"  Matrix is now unsorted but flag still says: {E.has_canonical_format}")

# Test 5: Check if has_sorted_indices flag exists and behaves similarly
print("\n" + "=" * 60)
print("Testing has_sorted_indices flag")
print("=" * 60)

F = sp.coo_matrix((data4, (row4, col4)), shape=(3, 3))
if hasattr(F, 'has_sorted_indices'):
    print(f"Initial has_sorted_indices: {F.has_sorted_indices}")
    F.sum_duplicates()
    print(f"After sum_duplicates: has_sorted_indices = {F.has_sorted_indices}")
    if len(F.row) > 1:
        F.row[0] = 2
        print(f"After modifying row: has_sorted_indices = {F.has_sorted_indices}")
else:
    print("has_sorted_indices attribute not found")