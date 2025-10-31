import numpy as np

# Test case 1: Array with NaN should be equivalent to itself (reflexivity)
arr = np.array([np.nan])
result = np.array_equiv(arr, arr)
print(f"Test 1 - Self-equivalence with NaN:")
print(f"  arr = {arr}")
print(f"  np.array_equiv(arr, arr) = {result}")
print(f"  Expected: True (reflexivity property)")
print()

# Test case 2: Array without NaN works correctly
arr2 = np.array([1.0, 2.0])
result2 = np.array_equiv(arr2, arr2)
print(f"Test 2 - Self-equivalence without NaN:")
print(f"  arr2 = {arr2}")
print(f"  np.array_equiv(arr2, arr2) = {result2}")
print(f"  Expected: True (works correctly)")
print()

# Test case 3: Multiple NaN values
arr3 = np.array([np.nan, np.nan, 1.0])
result3 = np.array_equiv(arr3, arr3)
print(f"Test 3 - Self-equivalence with multiple NaN:")
print(f"  arr3 = {arr3}")
print(f"  np.array_equiv(arr3, arr3) = {result3}")
print(f"  Expected: True (reflexivity property)")