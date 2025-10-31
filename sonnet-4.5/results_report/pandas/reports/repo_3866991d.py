import numpy as np
from pandas.api import indexers

# Test case 1: Empty numpy float array (default dtype) - FAILS
print("Test 1: Empty numpy float array (default dtype)")
arr = np.array([1, 2, 3, 4, 5])
empty_float_arr = np.array([])  # Default dtype is float64
print(f"  empty_float_arr dtype: {empty_float_arr.dtype}")

try:
    result = indexers.check_array_indexer(arr, empty_float_arr)
    print(f"  Result: {result}")
except IndexError as e:
    print(f"  Error: {e}")

print()

# Test case 2: Empty Python list - WORKS
print("Test 2: Empty Python list")
try:
    result = indexers.check_array_indexer(arr, [])
    print(f"  Result: {result}")
    print(f"  Result dtype: {result.dtype}")
except IndexError as e:
    print(f"  Error: {e}")

print()

# Test case 3: Empty numpy integer array - WORKS
print("Test 3: Empty numpy integer array (explicit dtype)")
empty_int_arr = np.array([], dtype=np.int64)
print(f"  empty_int_arr dtype: {empty_int_arr.dtype}")

try:
    result = indexers.check_array_indexer(arr, empty_int_arr)
    print(f"  Result: {result}")
    print(f"  Result dtype: {result.dtype}")
except IndexError as e:
    print(f"  Error: {e}")

print()

# Test case 4: Empty boolean array - FAILS differently
print("Test 4: Empty numpy boolean array")
empty_bool_arr = np.array([], dtype=bool)
print(f"  empty_bool_arr dtype: {empty_bool_arr.dtype}")

try:
    result = indexers.check_array_indexer(arr, empty_bool_arr)
    print(f"  Result: {result}")
    print(f"  Result dtype: {result.dtype}")
except IndexError as e:
    print(f"  Error: {e}")