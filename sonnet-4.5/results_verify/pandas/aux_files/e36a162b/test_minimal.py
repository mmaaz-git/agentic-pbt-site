import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.core.ops.mask_ops import kleene_and, kleene_or, kleene_xor

print("Testing the exact condition that causes recursion:")
print("=" * 50)

# According to the bug report, both masks being None causes recursion
arr1 = np.array([True, False], dtype=bool)
arr2 = np.array([True, True], dtype=bool)

print("\n1. Both masks are None:")
print(f"   left_mask=None, right_mask=None")
try:
    result = kleene_and(arr1, arr2, None, None)
    print(f"   Result: {result}")
except RecursionError as e:
    print(f"   -> RecursionError: Yes, infinite recursion occurs")

print("\n2. Only left_mask is None:")
print(f"   left_mask=None, right_mask=np.array([False, False])")
mask2 = np.array([False, False], dtype=bool)
try:
    result = kleene_and(arr1, arr2, None, mask2)
    print(f"   Result: {result}")
except RecursionError as e:
    print(f"   -> RecursionError occurred")

print("\n3. Only right_mask is None:")
print(f"   left_mask=np.array([False, False]), right_mask=None")
mask1 = np.array([False, False], dtype=bool)
try:
    result = kleene_and(arr1, arr2, mask1, None)
    print(f"   Result: {result}")
except RecursionError as e:
    print(f"   -> RecursionError occurred")

print("\n4. Both masks are present:")
print(f"   left_mask=np.array([False, False]), right_mask=np.array([False, False])")
try:
    result = kleene_and(arr1, arr2, mask1, mask2)
    print(f"   Result: {result}")
except RecursionError as e:
    print(f"   -> RecursionError occurred")

# Now check the logic in the code
print("\n" + "=" * 50)
print("Looking at the recursion logic in kleene_and:")
print("Line 156-157: if left_mask is None: return kleene_and(right, left, right_mask, left_mask)")
print("\nWhen both masks are None:")
print("1. left_mask=None, right_mask=None")
print("2. Condition 'if left_mask is None' is True")
print("3. Calls kleene_and(right, left, right_mask=None, left_mask=None)")
print("4. In the recursive call: left_mask=None again (was right_mask)")
print("5. Condition 'if left_mask is None' is True again")
print("6. Infinite recursion!")
print("\nThe bug report is correct about the infinite recursion.")