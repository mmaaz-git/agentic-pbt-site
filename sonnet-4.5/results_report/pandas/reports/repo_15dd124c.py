from pandas.arrays import SparseArray
import numpy as np

# Create a simple SparseArray with integer values
# Integer arrays default to fill_value=0
sparse = SparseArray([1, 2, 3])

print(f"SparseArray: {sparse}")
print(f"Fill value: {sparse.fill_value}")
print(f"_null_fill_value: {sparse._null_fill_value}")

# This should calculate cumulative sum [1, 3, 6]
# But it will cause RecursionError
try:
    result = sparse.cumsum()
    print(f"Cumsum result: {result}")
except RecursionError as e:
    print(f"\nRecursionError occurred!")
    print(f"Error: {e}")