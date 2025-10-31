import numpy as np
from pandas.arrays import SparseArray

sparse = SparseArray([0, 0, 1, 2], fill_value=0)
print("Original SparseArray:", type(sparse))
print()

# What the documentation says to use for converting to dense array
result_numpy = np.asarray(sparse, dtype=np.float64)
print("np.asarray(sparse, dtype=np.float64):")
print(f"  Type: {type(result_numpy)}")
print(f"  Value: {result_numpy}")
print()

# What astype currently does
result_astype = sparse.astype(np.float64)
print("sparse.astype(np.float64):")
print(f"  Type: {type(result_astype)}")
print(f"  Value: {result_astype}")
print()

# Check if they're identical
print("Are they identical?", np.array_equal(result_numpy, result_astype))