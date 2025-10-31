import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from xarray.core.nputils import inverse_permutation

indices = np.array([1], dtype=np.intp)
N = 2

print("Testing inverse_permutation with partial permutation:")
print(f"Initial indices: {indices}")
print(f"N (size of output array): {N}")
print()

inv1 = inverse_permutation(indices, N)
print(f"First inverse (inv1): {inv1}")
print(f"Note: inv1[0] = -1 (sentinel value indicating position 0 not in original permutation)")
print(f"      inv1[1] = 0 (element at position 0 in original goes to position 1)")
print()

inv2 = inverse_permutation(inv1, N)
print(f"Second inverse (inv2): {inv2}")
print(f"Note: When inv1=[-1, 0] is used as input:")
print(f"      The -1 is treated as a valid index by numpy (negative indexing)")
print(f"      inv2[-1] gets set to 0 (i.e., inv2[1] = 0)")
print(f"      inv2[0] gets set to 1")
print()

print(f"Expected result: {indices} (should be involutive)")
print(f"Actual result: {inv2}")
print(f"Are they equal? {np.array_equal(inv2, indices)}")
print()

print("Why this is wrong:")
print("1. The function uses -1 as a sentinel value to mark 'not in permutation'")
print("2. But it doesn't validate that input indices are non-negative")
print("3. When -1 appears in input, numpy treats it as index from end (last element)")
print("4. This breaks the mathematical property of involution for partial permutations")