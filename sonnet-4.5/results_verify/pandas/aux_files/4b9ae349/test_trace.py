import numpy as np
from pandas.core.util.hashing import hash_array

# Test with numeric arrays
print("Testing numeric array with different hash_keys:")
arr = np.array([1, 2, 3], dtype=np.int64)
hash1 = hash_array(arr, hash_key="0123456789123456")
hash2 = hash_array(arr, hash_key="AAAAAAAAAAAAAAAA")
print(f"  With default key: {hash1}")
print(f"  With custom key:  {hash2}")
print(f"  Are they equal? {np.array_equal(hash1, hash2)}")

# Test with object arrays
print("\nTesting object array with different hash_keys:")
obj_arr = np.array(['a', 'b', 'c'], dtype=object)
obj_hash1 = hash_array(obj_arr, hash_key="0123456789123456", categorize=False)
obj_hash2 = hash_array(obj_arr, hash_key="AAAAAAAAAAAAAAAA", categorize=False)
print(f"  With default key: {obj_hash1}")
print(f"  With custom key:  {obj_hash2}")
print(f"  Are they equal? {np.array_equal(obj_hash1, obj_hash2)}")

# Test with floats
print("\nTesting float array with different hash_keys:")
float_arr = np.array([1.0, 2.0, 3.0])
float_hash1 = hash_array(float_arr, hash_key="0123456789123456")
float_hash2 = hash_array(float_arr, hash_key="AAAAAAAAAAAAAAAA")
print(f"  With default key: {float_hash1}")
print(f"  With custom key:  {float_hash2}")
print(f"  Are they equal? {np.array_equal(float_hash1, float_hash2)}")

# Test with datetime
print("\nTesting datetime array with different hash_keys:")
dt_arr = np.array(['2021-01-01', '2021-01-02'], dtype='datetime64[D]')
dt_hash1 = hash_array(dt_arr, hash_key="0123456789123456")
dt_hash2 = hash_array(dt_arr, hash_key="AAAAAAAAAAAAAAAA")
print(f"  With default key: {dt_hash1}")
print(f"  With custom key:  {dt_hash2}")
print(f"  Are they equal? {np.array_equal(dt_hash1, dt_hash2)}")

# Test with bools
print("\nTesting boolean array with different hash_keys:")
bool_arr = np.array([True, False, True])
bool_hash1 = hash_array(bool_arr, hash_key="0123456789123456")
bool_hash2 = hash_array(bool_arr, hash_key="AAAAAAAAAAAAAAAA")
print(f"  With default key: {bool_hash1}")
print(f"  With custom key:  {bool_hash2}")
print(f"  Are they equal? {np.array_equal(bool_hash1, bool_hash2)}")