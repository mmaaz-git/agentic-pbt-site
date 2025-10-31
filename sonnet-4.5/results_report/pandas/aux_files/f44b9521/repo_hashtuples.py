from pandas.core.util.hashing import hash_tuples, hash_array
import numpy as np

# First show that hash_array handles empty input gracefully
empty_arr = np.array([], dtype=np.int64)
hash_arr_result = hash_array(empty_arr)
print(f"hash_array([]) works: {hash_arr_result}")
print(f"hash_array([]) dtype: {hash_arr_result.dtype}")

# Now show that hash_tuples crashes with empty input
print("\nCalling hash_tuples([])...")
try:
    result = hash_tuples([])
    print(f"hash_tuples([]) result: {result}")
except Exception as e:
    print(f"hash_tuples([]) raised {e.__class__.__name__}: {e}")