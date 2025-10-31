import numpy as np

# Test array_equal with NaN
arr = np.array([np.nan])
print("Testing array_equal with NaN:")
print(f"arr = {arr}")
print(f"np.array_equal(arr, arr) = {np.array_equal(arr, arr)}")
print(f"np.array_equal(arr, arr, equal_nan=False) = {np.array_equal(arr, arr, equal_nan=False)}")
print(f"np.array_equal(arr, arr, equal_nan=True) = {np.array_equal(arr, arr, equal_nan=True)}")

# Check array_equal documentation
print("\n" + "="*50)
print("array_equal documentation on equal_nan parameter:")
help(np.array_equal)