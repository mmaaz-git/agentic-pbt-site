import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.core.util.hashing import hash_tuples, hash_array
import numpy as np

print("Testing hash_array with empty input:")
empty_arr = np.array([], dtype=np.int64)
hash_arr_result = hash_array(empty_arr)
print(f"hash_array([]) works: {hash_arr_result}")
print(f"hash_array result type: {type(hash_arr_result)}")
print(f"hash_array result dtype: {hash_arr_result.dtype if hasattr(hash_arr_result, 'dtype') else 'N/A'}")

print("\nTesting hash_tuples with empty input:")
try:
    result = hash_tuples([])
    print(f"hash_tuples([]) works: {result}")
except Exception as e:
    print(f"hash_tuples([]) failed with {type(e).__name__}: {e}")