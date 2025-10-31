import numpy as np
import pandas.arrays as pa

# What should cumsum produce?
dense = np.array([1, 0, 2, 0, 3])
print(f"Dense array: {dense}")
print(f"NumPy cumsum: {dense.cumsum()}")

# Test with a working sparse array (NaN fill_value)
sparse_nan = pa.SparseArray([1.0, np.nan, 2.0, np.nan, 3.0], fill_value=np.nan)
print(f"\nSparse with NaN fill: {sparse_nan}")
result_nan = sparse_nan.cumsum()
print(f"Cumsum result: {result_nan}")

# What does documentation say about cumsum behavior?
print("\n" + "="*50)
print("Documentation says:")
print("- Cumulative sum of non-NA/null values")
print("- Non-NA/null values will be skipped")
print("- Result will preserve NaN locations")
print("- Fill value will be np.nan regardless")
print("="*50)