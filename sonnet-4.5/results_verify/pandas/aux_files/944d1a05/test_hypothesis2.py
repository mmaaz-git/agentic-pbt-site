import numpy as np
from pandas.core.util.hashing import hash_array

# Manual test of the hypothesis test logic with specific inputs
str_list = ['']
hash_key1 = '000000000000000\x80'

print(f"Testing with str_list={str_list}, hash_key1={repr(hash_key1)}")
print(f"hash_key1 length: {len(hash_key1)} characters")
print(f"hash_key1 encoded length: {len(hash_key1.encode('utf8'))} bytes")

arr = np.array(str_list, dtype=object)
hash_key2 = hash_key1[:15] + ('x' if hash_key1[15] != 'x' else 'y')

print(f"hash_key2={repr(hash_key2)}")
print(f"hash_key2 length: {len(hash_key2)} characters")
print(f"hash_key2 encoded length: {len(hash_key2.encode('utf8'))} bytes")

try:
    result1 = hash_array(arr, hash_key=hash_key1)
    print(f"result1: {result1}")
except Exception as e:
    print(f"hash_key1 failed with: {type(e).__name__}: {e}")

try:
    result2 = hash_array(arr, hash_key=hash_key2)
    print(f"result2: {result2}")
except Exception as e:
    print(f"hash_key2 failed with: {type(e).__name__}: {e}")