#!/usr/bin/env python3

import numpy as np
from pandas.core.util.hashing import hash_array, combine_hash_arrays

print("Testing manual reproduction of the bug:")
print("=" * 50)

arr1 = np.array([0], dtype=np.int64)
arr2 = np.array([0, 0], dtype=np.int64)

print(f"arr1: {arr1} (length: {len(arr1)})")
print(f"arr2: {arr2} (length: {len(arr2)})")

hash1 = hash_array(arr1)
hash2 = hash_array(arr2)

print(f"hash1: {hash1} (length: {len(hash1)})")
print(f"hash2: {hash2} (length: {len(hash2)})")

try:
    result = combine_hash_arrays(iter([hash1, hash2]), 2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")