import numpy as np
from pandas.core.util.hashing import hash_array

# Test case demonstrating the bug
arr_pos = np.array([0.0])
arr_neg = np.array([-0.0])

print("=== Testing hash_array with signed zeros ===")
print()
print(f"arr_pos = np.array([0.0])")
print(f"arr_neg = np.array([-0.0])")
print()
print(f"Arrays equal (np.array_equal): {np.array_equal(arr_pos, arr_neg)}")
print(f"Element equal (0.0 == -0.0): {0.0 == -0.0}")
print()
print(f"hash_array([0.0]):  {hash_array(arr_pos)}")
print(f"hash_array([-0.0]): {hash_array(arr_neg)}")
print()
print(f"Hashes equal: {np.array_equal(hash_array(arr_pos), hash_array(arr_neg))}")
print()
print("This violates the hash invariant: if a == b, then hash(a) must equal hash(b)")
print()
print("For comparison, Python's built-in hash function handles this correctly:")
print(f"hash(0.0):  {hash(0.0)}")
print(f"hash(-0.0): {hash(-0.0)}")
print(f"hash(0.0) == hash(-0.0): {hash(0.0) == hash(-0.0)}")