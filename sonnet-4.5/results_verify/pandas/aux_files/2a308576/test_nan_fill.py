import pandas.arrays as pa
import numpy as np

# Test with NaN fill value
arr_nan = pa.SparseArray([1, np.nan, 2, np.nan, 3], fill_value=np.nan)
print(f"Array with NaN fill_value: {arr_nan}")
print(f"Fill value: {arr_nan.fill_value}")

try:
    result_nan = arr_nan.cumsum()
    print(f"Cumsum result: {result_nan}")
    print(f"Result fill_value: {result_nan.fill_value}")
except Exception as e:
    print(f"Error with NaN fill_value: {e}")

print("\n" + "="*50 + "\n")

# Test with non-NaN fill value (0)
arr_zero = pa.SparseArray([1, 0, 2, 0, 3], fill_value=0)
print(f"Array with 0 fill_value: {arr_zero}")
print(f"Fill value: {arr_zero.fill_value}")

try:
    result_zero = arr_zero.cumsum()
    print(f"Cumsum result: {result_zero}")
    print(f"Result fill_value: {result_zero.fill_value}")
except RecursionError:
    print("RecursionError with non-NaN fill_value")
except Exception as e:
    print(f"Other error with 0 fill_value: {e}")