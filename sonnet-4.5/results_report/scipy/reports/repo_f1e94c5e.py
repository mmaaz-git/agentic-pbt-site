import scipy.sparse as sp

# Create a CSR matrix
csr = sp.random(5, 5, density=0.3, format='csr', random_state=42)

# Convert CSR to COO
coo = csr.tocoo()

# Check for duplicates
positions = list(zip(coo.row, coo.col))
unique_positions = set(positions)

print(f"has_canonical_format: {coo.has_canonical_format}")
print(f"Total entries: {len(positions)}")
print(f"Unique positions: {len(unique_positions)}")
print(f"Has duplicates: {len(positions) != len(unique_positions)}")

# Verify indices are sorted in row-major order
is_sorted = all(positions[i] <= positions[i+1] for i in range(len(positions)-1))
print(f"Indices sorted in row-major order: {is_sorted}")

# Show that the matrix actually meets canonical format requirements
print(f"\nMatrix meets canonical requirements:")
print(f"  - No duplicates: {len(positions) == len(unique_positions)}")
print(f"  - Sorted indices: {is_sorted}")
print(f"  - has_canonical_format flag: {coo.has_canonical_format}")
print(f"\nConclusion: Flag is incorrectly set to {coo.has_canonical_format} when it should be True")