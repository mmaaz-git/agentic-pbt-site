import numpy as np
from pandas.arrays import SparseArray

# Test the reproduction code
empty_sparse = SparseArray([])
print(f"len: {len(empty_sparse)}")
print(f"npoints: {empty_sparse.npoints}")

density = empty_sparse.density
print(f"density: {density}")
print(f"Is NaN: {np.isnan(density)}")