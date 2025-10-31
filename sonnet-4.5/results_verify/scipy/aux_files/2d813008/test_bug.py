import numpy as np
from scipy import sparse

print("=" * 60)
print("Testing the reported bug scenario")
print("=" * 60)

# Test case from the bug report
data = [1.0]
rows = [0]
cols = [0]
shape = (100, 100)

print("\nCreating COO matrix with single element at (0, 0):")
original_coo = sparse.coo_matrix((data, (rows, cols)), shape=shape)
print(f"Original COO has_canonical_format: {original_coo.has_canonical_format}")

print("\nConverting COO -> CSR:")
csr = original_coo.tocsr()
print(f"CSR matrix created")

print("\nConverting CSR -> COO:")
coo_from_csr = csr.tocoo()
print(f"COO from CSR has_canonical_format: {coo_from_csr.has_canonical_format}")

print("\nChecking for actual duplicates in the converted COO:")
coords = list(zip(coo_from_csr.row.tolist(), coo_from_csr.col.tolist()))
has_duplicates = len(coords) != len(set(coords))
print(f"Number of coordinates: {len(coords)}")
print(f"Number of unique coordinates: {len(set(coords))}")
print(f"Actually has duplicates: {has_duplicates}")

print("\n" + "=" * 60)
print("Testing with multiple non-duplicate elements")
print("=" * 60)

# Test with multiple unique elements
data2 = [1.0, 2.0, 3.0]
rows2 = [0, 1, 2]
cols2 = [0, 1, 2]

print("\nCreating COO matrix with 3 diagonal elements:")
coo2 = sparse.coo_matrix((data2, (rows2, cols2)), shape=shape)
print(f"Original COO has_canonical_format: {coo2.has_canonical_format}")

csr2 = coo2.tocsr()
coo_from_csr2 = csr2.tocoo()
print(f"COO from CSR has_canonical_format: {coo_from_csr2.has_canonical_format}")

coords2 = list(zip(coo_from_csr2.row.tolist(), coo_from_csr2.col.tolist()))
has_duplicates2 = len(coords2) != len(set(coords2))
print(f"Actually has duplicates: {has_duplicates2}")

print("\n" + "=" * 60)
print("Testing with duplicate coordinates")
print("=" * 60)

# Test with actual duplicates
data3 = [1.0, 2.0, 3.0]
rows3 = [0, 0, 1]
cols3 = [0, 0, 1]

print("\nCreating COO matrix with duplicates at (0,0):")
coo3 = sparse.coo_matrix((data3, (rows3, cols3)), shape=shape)
print(f"Original COO has_canonical_format: {coo3.has_canonical_format}")

# Check the data before conversion
print(f"Original COO nnz (non-zero entries): {coo3.nnz}")
print(f"Original COO data: {coo3.data}")
print(f"Original COO row: {coo3.row}")
print(f"Original COO col: {coo3.col}")

csr3 = coo3.tocsr()
print(f"\nCSR nnz: {csr3.nnz}")

coo_from_csr3 = csr3.tocoo()
print(f"COO from CSR has_canonical_format: {coo_from_csr3.has_canonical_format}")
print(f"COO from CSR nnz: {coo_from_csr3.nnz}")
print(f"COO from CSR data: {coo_from_csr3.data}")
print(f"COO from CSR row: {coo_from_csr3.row}")
print(f"COO from CSR col: {coo_from_csr3.col}")

coords3 = list(zip(coo_from_csr3.row.tolist(), coo_from_csr3.col.tolist()))
has_duplicates3 = len(coords3) != len(set(coords3))
print(f"Actually has duplicates: {has_duplicates3}")

print("\n" + "=" * 60)
print("Testing has_canonical_format behavior")
print("=" * 60)

# Test what makes a COO matrix have canonical format
print("\nTesting fresh COO matrix without duplicates:")
coo_test = sparse.coo_matrix((data, (rows, cols)), shape=(10, 10))
print(f"Fresh COO has_canonical_format: {coo_test.has_canonical_format}")

print("\nTesting after sum_duplicates():")
coo_test.sum_duplicates()
print(f"After sum_duplicates() has_canonical_format: {coo_test.has_canonical_format}")

print("\nTesting COO created with duplicate coordinates:")
coo_dup = sparse.coo_matrix(([1, 2], ([0, 0], [0, 0])), shape=(10, 10))
print(f"COO with duplicates has_canonical_format: {coo_dup.has_canonical_format}")
coo_dup.sum_duplicates()
print(f"After sum_duplicates() has_canonical_format: {coo_dup.has_canonical_format}")