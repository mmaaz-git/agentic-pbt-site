import numpy as np
from pandas.core.util.hashing import hash_array

# Create a simple numeric array
arr = np.array([1, 2, 3])

# Hash with two different keys
hash1 = hash_array(arr, hash_key='0' * 16)
hash2 = hash_array(arr, hash_key='1' * 16)

print("Array:", arr)
print("Hash with key '0'*16:", hash1)
print("Hash with key '1'*16:", hash2)
print("Are they equal?:", np.array_equal(hash1, hash2))

# Additional test with different array types
print("\nTesting with different numeric types:")

# Integer array
int_arr = np.array([42])
int_hash1 = hash_array(int_arr, hash_key='0' * 16)
int_hash2 = hash_array(int_arr, hash_key='1' * 16)
print(f"Integer array {int_arr}: hashes equal? {np.array_equal(int_hash1, int_hash2)}")

# Float array
float_arr = np.array([3.14])
float_hash1 = hash_array(float_arr, hash_key='0' * 16)
float_hash2 = hash_array(float_arr, hash_key='1' * 16)
print(f"Float array {float_arr}: hashes equal? {np.array_equal(float_hash1, float_hash2)}")

# Boolean array
bool_arr = np.array([True, False])
bool_hash1 = hash_array(bool_arr, hash_key='0' * 16)
bool_hash2 = hash_array(bool_arr, hash_key='1' * 16)
print(f"Boolean array {bool_arr}: hashes equal? {np.array_equal(bool_hash1, bool_hash2)}")

# String array (object dtype) - this should work correctly
str_arr = np.array(['hello', 'world'], dtype=object)
str_hash1 = hash_array(str_arr, hash_key='0' * 16)
str_hash2 = hash_array(str_arr, hash_key='1' * 16)
print(f"String array {str_arr}: hashes equal? {np.array_equal(str_hash1, str_hash2)}")