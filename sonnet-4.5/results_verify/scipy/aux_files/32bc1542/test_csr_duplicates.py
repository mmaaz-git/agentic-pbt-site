import scipy.sparse as sp
import numpy as np

# Test 1: Create CSR with duplicates
row = np.array([0, 1, 2, 0])
col = np.array([0, 1, 1, 0])
data = np.array([1, 2, 4, 8])
csr = sp.csr_matrix((data, (row, col)), shape=(3, 3))
print("CSR matrix with duplicates in constructor:")
print("Array representation:", csr.toarray())
print("CSR.data:", csr.data)
print("CSR.indices:", csr.indices)
print()

# Test 2: Check if CSR from random has duplicates
csr2 = sp.random(5, 5, density=0.3, format='csr', random_state=42)
print("CSR from random:")
print("CSR has_canonical_format (if exists):", hasattr(csr2, 'has_canonical_format'))

# Test 3: Convert CSR to COO and check
coo = csr.tocoo()
print("\nCSR (with duplicates summed) to COO:")
print("COO.row:", coo.row)
print("COO.col:", coo.col)
print("COO.data:", coo.data)
print("COO.has_canonical_format:", coo.has_canonical_format)

# Check for duplicates in COO
positions = list(zip(coo.row, coo.col))
unique_positions = set(positions)
print(f"Total positions: {len(positions)}, Unique positions: {len(unique_positions)}")
print(f"Has duplicates: {len(positions) != len(unique_positions)}")