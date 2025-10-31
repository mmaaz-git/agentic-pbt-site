import numpy as np
from pandas.core.dtypes.dtypes import SparseDtype

dtype1 = SparseDtype(np.float32, np.nan)
dtype2 = SparseDtype(np.float32, 0.0)

print(f"dtype1 == dtype2: {dtype1 == dtype2}")
print(f"dtype2 == dtype1: {dtype2 == dtype1}")

# Also test the hash consistency claim
if dtype1 == dtype2:
    print(f"dtype1 hash: {hash(dtype1)}")
    print(f"dtype2 hash: {hash(dtype2)}")
    if hash(dtype1) != hash(dtype2):
        print("WARNING: Equal objects have different hashes!")