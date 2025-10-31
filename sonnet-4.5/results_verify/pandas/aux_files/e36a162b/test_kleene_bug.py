import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.core.ops.mask_ops import kleene_and, kleene_or, kleene_xor

# Test case 1: Reproduce the bug with kleene_and
print("Test 1: Testing kleene_and with both masks as None")
arr1 = np.array([True, False], dtype=bool)
arr2 = np.array([True, True], dtype=bool)

try:
    result = kleene_and(arr1, arr2, None, None)
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError occurred: {str(e)[:100]}...")
except Exception as e:
    print(f"Other error occurred: {type(e).__name__}: {e}")

# Test case 2: Test kleene_or with both masks as None
print("\nTest 2: Testing kleene_or with both masks as None")
try:
    result = kleene_or(arr1, arr2, None, None)
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError occurred: {str(e)[:100]}...")
except Exception as e:
    print(f"Other error occurred: {type(e).__name__}: {e}")

# Test case 3: Test kleene_xor with both masks as None
print("\nTest 3: Testing kleene_xor with both masks as None")
try:
    result = kleene_xor(arr1, arr2, None, None)
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError occurred: {str(e)[:100]}...")
except Exception as e:
    print(f"Other error occurred: {type(e).__name__}: {e}")

# Test case 4: Test with one mask being None (should work according to docs)
print("\nTest 4: Testing kleene_and with left_mask as None (should work)")
mask2 = np.array([False, False], dtype=bool)
try:
    result, result_mask = kleene_and(arr1, arr2, None, mask2)
    print(f"Result: {result}, Mask: {result_mask}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# Test case 5: Test with right_mask as None (should work)
print("\nTest 5: Testing kleene_and with right_mask as None (should work)")
mask1 = np.array([False, False], dtype=bool)
try:
    result, result_mask = kleene_and(arr1, arr2, mask1, None)
    print(f"Result: {result}, Mask: {result_mask}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")