import numpy as np

# Test what np.sum returns for empty arrays
empty_list = []
arr = np.asarray(empty_list, dtype=object)
print(f"np.asarray([], dtype=object): {repr(arr)}")
print(f"Type: {type(arr)}")
print(f"Shape: {arr.shape}")

result = np.sum(arr, axis=0)
print(f"\nnp.sum(arr, axis=0): {repr(result)}")
print(f"Type: {type(result)}")

# Test with list_with_sep calculation for empty list
list_of_columns = []
sep = ","
list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
print(f"\nlist_with_sep for empty list: {list_with_sep}")
print(f"Length calculation: 2 * 0 - 1 = {2 * len(list_of_columns) - 1}")

# What happens with negative index?
test_list = []
test_list[::2] = []
print(f"After setting test_list[::2] = []: {test_list}")