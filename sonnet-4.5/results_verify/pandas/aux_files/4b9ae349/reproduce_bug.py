import numpy as np
from pandas.core.util.hashing import hash_array

arr = np.array([1, 2, 3, 4, 5])

hash1 = hash_array(arr, hash_key="0123456789123456")
hash2 = hash_array(arr, hash_key="AAAAAAAAAAAAAAAA")
hash3 = hash_array(arr, hash_key="different_key123")

print(f"hash_key='0123456789123456': {hash1}")
print(f"hash_key='AAAAAAAAAAAAAAAA': {hash2}")
print(f"hash_key='different_key123': {hash3}")
print(f"\nAll identical: {np.array_equal(hash1, hash2) and np.array_equal(hash2, hash3)}")

obj_arr = np.array(['a', 'b'], dtype=object)
obj_hash1 = hash_array(obj_arr, hash_key="0123456789123456", categorize=False)
obj_hash2 = hash_array(obj_arr, hash_key="AAAAAAAAAAAAAAAA", categorize=False)
print(f"\nObject arrays respect hash_key: {not np.array_equal(obj_hash1, obj_hash2)}")