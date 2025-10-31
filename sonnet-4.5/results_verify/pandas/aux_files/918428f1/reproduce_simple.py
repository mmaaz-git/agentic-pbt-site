import numpy as np
from pandas.core.util.hashing import hash_array

arr = np.array([1, 2, 3])

hash1 = hash_array(arr, hash_key='0' * 16)
hash2 = hash_array(arr, hash_key='1' * 16)

print("Hash with key '0'*16:", hash1)
print("Hash with key '1'*16:", hash2)
print("Are they equal?:", np.array_equal(hash1, hash2))