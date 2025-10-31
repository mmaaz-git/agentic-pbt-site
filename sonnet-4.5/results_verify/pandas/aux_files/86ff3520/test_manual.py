import numpy as np
from pandas.core.util.hashing import hash_array

arr_pos = np.array([0.0])
arr_neg = np.array([-0.0])

print(f"Arrays equal: {np.array_equal(arr_pos, arr_neg)}")
print(f"hash_array([0.0]):  {hash_array(arr_pos)}")
print(f"hash_array([-0.0]): {hash_array(arr_neg)}")
print(f"Hashes equal: {np.array_equal(hash_array(arr_pos), hash_array(arr_neg))}")

# Let's also test Python's behavior
print("\nPython's behavior:")
print(f"0.0 == -0.0: {0.0 == -0.0}")
print(f"hash(0.0): {hash(0.0)}")
print(f"hash(-0.0): {hash(-0.0)}")
print(f"hash(0.0) == hash(-0.0): {hash(0.0) == hash(-0.0)}")