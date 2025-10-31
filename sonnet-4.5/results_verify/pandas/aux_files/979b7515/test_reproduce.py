import numpy as np
from pandas.arrays import SparseArray

# First, let's reproduce the exact bug
arr = SparseArray([0, 1, 2], fill_value=0)
result = arr.astype(np.float64)

print(f"Type of result: {type(result)}")
print(f"Result is SparseArray: {isinstance(result, SparseArray)}")
print(f"Result: {result}")