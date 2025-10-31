import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.core.util.hashing import hash_tuples, hash_array, combine_hash_arrays
import numpy as np
import itertools

# Test combine_hash_arrays with empty iterator
print("Testing combine_hash_arrays with empty iterator:")
empty_iter = iter([])
result = combine_hash_arrays(empty_iter, 0)
print(f"Result: {result}")
print(f"Type: {type(result)}")
print(f"Dtype: {result.dtype}")
print(f"Shape: {result.shape}")

# Test hash_array behavior with empty arrays of different types
print("\nTesting hash_array with various empty inputs:")

# Test with empty numpy array
empty_np = np.array([])
print(f"empty numpy array dtype: {empty_np.dtype}")
try:
    result = hash_array(empty_np)
    print(f"hash_array(np.array([])) succeeded: {result}")
except Exception as e:
    print(f"hash_array(np.array([])) failed: {e}")

# Test with empty int64 array
empty_int64 = np.array([], dtype=np.int64)
print(f"\nempty int64 array dtype: {empty_int64.dtype}")
result = hash_array(empty_int64)
print(f"hash_array(np.array([], dtype=np.int64)) succeeded: {result}")
print(f"Result dtype: {result.dtype}")
print(f"Result shape: {result.shape}")