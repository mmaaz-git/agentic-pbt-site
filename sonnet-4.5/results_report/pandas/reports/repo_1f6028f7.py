import numpy as np
from pandas.arrays import SparseArray

# Create a SparseArray with NaN values and a non-null fill_value
arr = SparseArray([1.0, np.nan, 0.0], fill_value=0.0)

# Test max with skipna=False
max_result = arr.max(skipna=False)
print(f"SparseArray.max(skipna=False): {max_result}")
print(f"Expected: nan")
print()

# Test min with skipna=False
min_result = arr.min(skipna=False)
print(f"SparseArray.min(skipna=False): {min_result}")
print(f"Expected: nan")
print()

# Compare with NumPy behavior
numpy_array = np.array([1.0, np.nan, 0.0])
numpy_max = np.max(numpy_array)
numpy_min = np.min(numpy_array)
print(f"NumPy max: {numpy_max}")
print(f"NumPy min: {numpy_min}")
print()

# Compare with dense pandas Series
import pandas as pd
series = pd.Series([1.0, np.nan, 0.0])
series_max = series.max(skipna=False)
series_min = series.min(skipna=False)
print(f"Series.max(skipna=False): {series_max}")
print(f"Series.min(skipna=False): {series_min}")