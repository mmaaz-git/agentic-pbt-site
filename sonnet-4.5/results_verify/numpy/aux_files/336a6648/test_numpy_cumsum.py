import numpy as np

# Test numpy's cumsum behavior
arr = np.array([1, 2, 3])
print(f"Array: {arr}")
print(f"Cumsum: {np.cumsum(arr)}")

# Test with zeros
arr_zeros = np.array([0, 1, 0, 2, 0])
print(f"\nArray with zeros: {arr_zeros}")
print(f"Cumsum: {np.cumsum(arr_zeros)}")

# Check numpy cumsum documentation
print("\nnumpy.cumsum docstring:")
print(np.cumsum.__doc__[:500])