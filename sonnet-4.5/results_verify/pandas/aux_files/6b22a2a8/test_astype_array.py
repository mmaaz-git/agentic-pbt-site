from pandas.core.dtypes.astype import astype_array
import numpy as np

# Test what astype_array actually returns
arr = np.array([1, 2, 3])
result = astype_array(arr, np.dtype(np.float64), copy=False)
print(f"Type of result from astype_array: {type(result)}")
print(f"Result: {result}")