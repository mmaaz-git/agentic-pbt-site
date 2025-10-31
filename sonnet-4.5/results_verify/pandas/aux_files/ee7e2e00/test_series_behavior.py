import pandas as pd
import numpy as np

print("Pandas Series behavior for arrays with all equal values:\n")

# Single element
s1 = pd.Series([0])
print(f"pd.Series([0]).argmin() = {s1.argmin()}")
print(f"pd.Series([0]).argmax() = {s1.argmax()}")

# Multiple equal elements
s2 = pd.Series([5, 5, 5, 5])
print(f"pd.Series([5, 5, 5, 5]).argmin() = {s2.argmin()}")
print(f"pd.Series([5, 5, 5, 5]).argmax() = {s2.argmax()}")

# Regular pandas array
arr = pd.array([0, 0, 0])
print(f"\npd.array([0, 0, 0]) type: {type(arr)}")
try:
    print(f"pd.array([0, 0, 0]).argmin() = {arr.argmin()}")
except Exception as e:
    print(f"Error: {e}")

# Check if SparseArray is supposed to be numpy-compatible
print("\nChecking SparseArray as an ExtensionArray:")
from pandas.arrays import SparseArray
print(f"Is SparseArray a pandas ExtensionArray? {isinstance(SparseArray([1,2,3]), pd.api.extensions.ExtensionArray)}")