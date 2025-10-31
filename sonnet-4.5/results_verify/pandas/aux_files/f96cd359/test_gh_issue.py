import numpy as np
import pandas as pd
from pandas.arrays import SparseArray
from pandas import SparseDtype

print(f"Pandas version: {pd.__version__}")

# Test the exact scenario from GH#34457
arr = SparseArray([1, 0, 0, 2])
print(f"\nOriginal sparse array dtype: {arr.dtype}")

# Try astype with float (non-SparseDtype)
result = arr.astype(float)
print(f"\nAfter astype(float):")
print(f"  Type of result: {type(result)}")
print(f"  Dtype of result: {result.dtype if hasattr(result, 'dtype') else 'N/A'}")
print(f"  Is SparseArray: {isinstance(result, SparseArray)}")

# Try astype with np.float64 (non-SparseDtype)
result2 = arr.astype(np.float64)
print(f"\nAfter astype(np.float64):")
print(f"  Type of result: {type(result2)}")
print(f"  Dtype of result: {result2.dtype if hasattr(result2, 'dtype') else 'N/A'}")
print(f"  Is SparseArray: {isinstance(result2, SparseArray)}")

# Try with SparseDtype explicitly
result3 = arr.astype(SparseDtype(float))
print(f"\nAfter astype(SparseDtype(float)):")
print(f"  Type of result: {type(result3)}")
print(f"  Dtype of result: {result3.dtype if hasattr(result3, 'dtype') else 'N/A'}")
print(f"  Is SparseArray: {isinstance(result3, SparseArray)}")