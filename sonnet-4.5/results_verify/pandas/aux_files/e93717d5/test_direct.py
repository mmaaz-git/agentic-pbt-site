import numpy as np
from pandas.api import indexers

print("=" * 60)
print("Testing empty float array (should fail according to bug report)")
print("=" * 60)
arr = np.array([1, 2, 3, 4, 5])
empty_float_arr = np.array([])

print(f"arr: {arr}")
print(f"empty_float_arr: {empty_float_arr}")
print(f"empty_float_arr dtype: {empty_float_arr.dtype}")
print(f"empty_float_arr shape: {empty_float_arr.shape}")

try:
    result = indexers.check_array_indexer(arr, empty_float_arr)
    print(f"Result: {result}")
except IndexError as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Testing empty Python list (should work according to bug report)")
print("=" * 60)
try:
    result = indexers.check_array_indexer(arr, [])
    print(f"Empty list result: {result}")
    print(f"Result dtype: {result.dtype}")
except Exception as e:
    print(f"Unexpected error: {e}")

print("\n" + "=" * 60)
print("Testing empty integer numpy array (should work according to bug report)")
print("=" * 60)
try:
    empty_int_arr = np.array([], dtype=np.int64)
    print(f"empty_int_arr dtype: {empty_int_arr.dtype}")
    result = indexers.check_array_indexer(arr, empty_int_arr)
    print(f"Empty int array result: {result}")
    print(f"Result dtype: {result.dtype}")
except Exception as e:
    print(f"Unexpected error: {e}")

print("\n" + "=" * 60)
print("Testing edge case: what about empty bool array?")
print("=" * 60)
try:
    empty_bool_arr = np.array([], dtype=bool)
    print(f"empty_bool_arr dtype: {empty_bool_arr.dtype}")
    result = indexers.check_array_indexer(arr, empty_bool_arr)
    print(f"Empty bool array result: {result}")
    print(f"Result dtype: {result.dtype}")
except Exception as e:
    print(f"Error: {e}")