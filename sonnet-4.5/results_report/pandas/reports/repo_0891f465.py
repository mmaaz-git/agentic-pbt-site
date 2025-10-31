import numpy as np
from pandas.core.util.hashing import hash_array

# Test with numeric arrays (int64)
arr = np.array([1, 2, 3, 4, 5])

hash1 = hash_array(arr, hash_key="0123456789123456")
hash2 = hash_array(arr, hash_key="AAAAAAAAAAAAAAAA")
hash3 = hash_array(arr, hash_key="different_key123")

print("Testing numeric arrays (int64):")
print(f"hash_key='0123456789123456': {hash1}")
print(f"hash_key='AAAAAAAAAAAAAAAA': {hash2}")
print(f"hash_key='different_key123': {hash3}")
print(f"All hashes identical: {np.array_equal(hash1, hash2) and np.array_equal(hash2, hash3)}")

# Test with object arrays
print("\nTesting object arrays:")
obj_arr = np.array(['a', 'b', 'c'], dtype=object)
obj_hash1 = hash_array(obj_arr, hash_key="0123456789123456", categorize=False)
obj_hash2 = hash_array(obj_arr, hash_key="AAAAAAAAAAAAAAAA", categorize=False)
print(f"hash_key='0123456789123456': {obj_hash1}")
print(f"hash_key='AAAAAAAAAAAAAAAA': {obj_hash2}")
print(f"Object arrays respect hash_key: {not np.array_equal(obj_hash1, obj_hash2)}")

# Test with other numeric types
print("\nTesting float arrays:")
float_arr = np.array([1.0, 2.0, 3.0])
float_hash1 = hash_array(float_arr, hash_key="0123456789123456")
float_hash2 = hash_array(float_arr, hash_key="AAAAAAAAAAAAAAAA")
print(f"Float arrays respect hash_key: {not np.array_equal(float_hash1, float_hash2)}")

print("\nTesting bool arrays:")
bool_arr = np.array([True, False, True])
bool_hash1 = hash_array(bool_arr, hash_key="0123456789123456")
bool_hash2 = hash_array(bool_arr, hash_key="AAAAAAAAAAAAAAAA")
print(f"Bool arrays respect hash_key: {not np.array_equal(bool_hash1, bool_hash2)}")

print("\nTesting datetime arrays:")
datetime_arr = np.array(['2021-01-01', '2021-01-02'], dtype='datetime64')
datetime_hash1 = hash_array(datetime_arr, hash_key="0123456789123456")
datetime_hash2 = hash_array(datetime_arr, hash_key="AAAAAAAAAAAAAAAA")
print(f"Datetime arrays respect hash_key: {not np.array_equal(datetime_hash1, datetime_hash2)}")