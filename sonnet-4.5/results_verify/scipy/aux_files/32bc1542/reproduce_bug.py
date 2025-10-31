import scipy.sparse as sp

csr = sp.random(5, 5, density=0.3, format='csr', random_state=42)
coo = csr.tocoo()

positions = list(zip(coo.row, coo.col))
unique_positions = set(positions)

print(f"has_canonical_format: {coo.has_canonical_format}")
print(f"Total entries: {len(positions)}")
print(f"Unique positions: {len(unique_positions)}")
print(f"Has duplicates: {len(positions) != len(unique_positions)}")