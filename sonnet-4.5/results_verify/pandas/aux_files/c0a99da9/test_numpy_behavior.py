import numpy as np

# Test what happens with kind=None vs kind='quicksort'
arr = np.array([3, 1, 2])

# Test with default (kind=None)
result1 = np.argsort(arr)
print(f"np.argsort([3, 1, 2]) with defaults: {result1}")

# Test with explicit kind=None
result2 = np.argsort(arr, kind=None)
print(f"np.argsort([3, 1, 2], kind=None): {result2}")

# Test with kind='quicksort'
result3 = np.argsort(arr, kind='quicksort')
print(f"np.argsort([3, 1, 2], kind='quicksort'): {result3}")

# Check if they're the same
print(f"\nAll results equal: {np.array_equal(result1, result2) and np.array_equal(result2, result3)}")

# Check numpy version
print(f"\nnumpy version: {np.__version__}")