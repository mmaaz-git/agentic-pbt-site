import numpy as np

# Test what the actual Python slice behavior is for negative steps
print("Testing actual Python/numpy slice behavior with negative steps:")

# Test 1: Simple array with negative step
arr = np.array([0, 1, 2, 3, 4])
slc = slice(None, None, -1)
result = arr[slc]
print(f"arr = {arr}")
print(f"arr[slice(None, None, -1)] = {result}")
print(f"len(arr[slice(None, None, -1)]) = {len(result)}")
print()

# Test 2: With start specified
arr = np.array([0, 1, 2, 3, 4])
slc = slice(3, None, -1)
result = arr[slc]
print(f"arr = {arr}")
print(f"arr[slice(3, None, -1)] = {result}")
print(f"len(arr[slice(3, None, -1)]) = {len(result)}")
print()

# Test 3: With stop specified
arr = np.array([0, 1, 2, 3, 4])
slc = slice(None, 1, -1)
result = arr[slc]
print(f"arr = {arr}")
print(f"arr[slice(None, 1, -1)] = {result}")
print(f"len(arr[slice(None, 1, -1)]) = {len(result)}")
print()

# Test 4: Length is always non-negative
print("Testing that len() always returns non-negative values:")
for length in [1, 2, 5, 10]:
    for step in [-1, -2, -3]:
        arr = np.arange(length)
        slc = slice(None, None, step)
        result_len = len(arr[slc])
        print(f"  len(np.arange({length})[slice(None, None, {step})]) = {result_len} (>= 0: {result_len >= 0})")