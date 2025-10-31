import scipy.sparse as sp
import numpy as np

# Test CSR canonical format
csr = sp.random(5, 5, density=0.3, format='csr', random_state=42)
print("CSR matrix:")
print("has_canonical_format:", csr.has_canonical_format)

# Convert to COO
coo = csr.tocoo()
print("\nAfter converting CSR to COO:")
print("COO has_canonical_format:", coo.has_canonical_format)

# Check if COO is actually sorted
is_sorted = all(coo.row[i] < coo.row[i+1] or
                (coo.row[i] == coo.row[i+1] and coo.col[i] <= coo.col[i+1])
                for i in range(len(coo.row)-1))
print("COO indices are sorted:", is_sorted)

# Check for duplicates
positions = list(zip(coo.row, coo.col))
unique_positions = set(positions)
has_duplicates = len(positions) != len(unique_positions)
print("COO has duplicates:", has_duplicates)

# So based on the canonical format definition:
# - No duplicates: True
# - Sorted indices: ?
print("\nBased on canonical format definition:")
print("- No duplicates:", not has_duplicates)
print("- Sorted indices:", is_sorted)
print("Should be canonical:", not has_duplicates and is_sorted)