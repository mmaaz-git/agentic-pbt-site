from pandas.arrays import SparseArray
import numpy as np

# Test case that should crash
sparse = SparseArray([0])  # Array containing only fill_value (default is 0)

print("Testing SparseArray([0]).argmin()...")
result = sparse.argmin()
print(f"Result: {result}")