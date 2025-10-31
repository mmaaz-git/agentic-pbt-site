import numpy as np
from pandas.arrays import SparseArray

# Create an empty sparse array
empty_sparse = SparseArray([])

# Print basic properties
print(f"len: {len(empty_sparse)}")
print(f"npoints: {empty_sparse.npoints}")

# Try to access the density property
try:
    density = empty_sparse.density
    print(f"density: {density}")
    print(f"Is NaN: {np.isnan(density)}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")