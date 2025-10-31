import numpy as np
from pandas.core.array_algos.masked_accumulations import cumsum

arr = np.array([10, 20, 30, 40, 50], dtype=np.int64)
mask = np.array([False, True, False, False, True], dtype=bool)

print("Before:", arr)
print("Original array id:", id(arr))

result_values, result_mask = cumsum(arr, mask, skipna=True)

print("After:", arr)
print("Result:", result_values)
print("Result array id:", id(result_values))
print("Arrays are same object:", arr is result_values)