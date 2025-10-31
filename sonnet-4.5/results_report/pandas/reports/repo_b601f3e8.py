import numpy as np
from pandas._libs import missing as libmissing
import pandas.core.ops as ops

# Test case 1: True ^ NA
left_true = np.array([True])
mask_false = np.array([False])

# Test case 2: NA ^ True
left_na = np.array([False])
mask_true = np.array([True])

# Execute operations
result1, mask1 = ops.kleene_xor(left_true, libmissing.NA, mask_false, None)
result2, mask2 = ops.kleene_xor(left_na, True, mask_true, None)

print(f'True ^ NA: result={result1[0]}, mask={mask1[0]}')
print(f'NA ^ True: result={result2[0]}, mask={mask2[0]}')

# Check if results are the same (they should be for commutative operations)
if result1[0] == result2[0]:
    print("✓ Results are equal (commutative)")
else:
    print("✗ Results differ - commutativity violated!")
    print(f"  True ^ NA returns result={result1[0]}")
    print(f"  NA ^ True returns result={result2[0]}")

# This assertion will fail, demonstrating the bug
assert result1[0] == result2[0], f"Commutativity violated: {result1[0]} != {result2[0]}"