import numpy as np
from pandas.core.arrays.sparse import SparseArray

# Test case where all values equal fill value
arr = np.array([0, 0])
sparse = SparseArray(arr)

print("Testing when all values equal fill_value:")
print(f"Array: {arr}")
print(f"Fill value: {sparse.fill_value}")
print(f"Sparse values (sp_values): {sparse.sp_values}")
print(f"Length of sp_values: {len(sparse.sp_values)}")
print()

# The problematic code is at line 1658:
# _candidate = non_nan_idx[func(non_nans)]
# When sp_values is empty, non_nans is also empty after filtering

# Let me also test what NumPy does in this case
print("NumPy behavior:")
print(f"np.argmin([0, 0]): {np.argmin([0, 0])}")
print(f"np.argmax([0, 0]): {np.argmax([0, 0])}")
print()

# Test with different all-same values
arr2 = np.array([5, 5, 5])
print(f"np.argmin([5, 5, 5]): {np.argmin(arr2)}")
print(f"np.argmax([5, 5, 5]): {np.argmax(arr2)}")
print()

# Test SparseArray with non-default fill value
sparse2 = SparseArray([5, 5, 5], fill_value=5)
print(f"SparseArray([5, 5, 5], fill_value=5)")
print(f"Sparse values: {sparse2.sp_values}")
try:
    print(f"sparse.argmin(): {sparse2.argmin()}")
except Exception as e:
    print(f"Error: {e}")

# Test SparseArray with some values not equal to fill
sparse3 = SparseArray([0, 1, 0])  # fill_value=0 by default
print(f"\nSparseArray([0, 1, 0])")
print(f"Sparse values: {sparse3.sp_values}")
print(f"sparse.argmin(): {sparse3.argmin()}")
print(f"sparse.argmax(): {sparse3.argmax()}")