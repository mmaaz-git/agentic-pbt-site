import numpy as np
from pandas.core.dtypes.dtypes import SparseDtype

dtype1 = SparseDtype(np.float32, np.nan)
dtype2 = SparseDtype(np.float32, 0.0)

print(f"dtype1 = {dtype1}")
print(f"dtype2 = {dtype2}")
print()
print(f"dtype1 == dtype2: {dtype1 == dtype2}")
print(f"dtype2 == dtype1: {dtype2 == dtype1}")
print()
print("This demonstrates asymmetric equality - dtype1 == dtype2 returns True")
print("but dtype2 == dtype1 returns False, violating the symmetric property of equality.")