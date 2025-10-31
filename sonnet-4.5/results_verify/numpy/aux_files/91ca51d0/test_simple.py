import numpy as np

# Test the simple reproduction case
arr = np.array([np.nan])
result = np.array_equiv(arr, arr)
print(f"np.array_equiv(arr, arr) = {result}")
print(f"Expected: True (reflexivity - an array should always be equivalent to itself)")

# Test with multiple NaN values
arr2 = np.array([np.nan, np.nan, 1.0])
result2 = np.array_equiv(arr2, arr2)
print(f"\nArray with multiple NaNs: {arr2}")
print(f"np.array_equiv(arr2, arr2) = {result2}")

# Test with no NaN values
arr3 = np.array([1.0, 2.0, 3.0])
result3 = np.array_equiv(arr3, arr3)
print(f"\nArray without NaN: {arr3}")
print(f"np.array_equiv(arr3, arr3) = {result3}")