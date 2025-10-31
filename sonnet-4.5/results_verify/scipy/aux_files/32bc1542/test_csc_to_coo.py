import scipy.sparse as sp

# Test CSC to COO conversion as well
csc = sp.random(5, 5, density=0.3, format='csc', random_state=42)
coo = csc.tocoo()

positions = list(zip(coo.row, coo.col))
unique_positions = set(positions)

print("CSC to COO conversion:")
print(f"has_canonical_format: {coo.has_canonical_format}")
print(f"Total entries: {len(positions)}")
print(f"Unique positions: {len(unique_positions)}")
print(f"Has duplicates: {len(positions) != len(unique_positions)}")