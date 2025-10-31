import numpy as np
import pandas as pd
from pandas.api.extensions import take

index = pd.Index([10.0, 20.0, 30.0])
arr = np.array([10.0, 20.0, 30.0])

index_result = take(index, [0, -1, 2], allow_fill=True, fill_value=None)
array_result = take(arr, [0, -1, 2], allow_fill=True, fill_value=None)

print("Index result:", list(index_result))
print("Array result:", list(array_result))

print(f"\nChecking array[1]: pd.isna({array_result[1]}) = {pd.isna(array_result[1])}")
print(f"Checking index[1]: pd.isna({index_result[1]}) = {pd.isna(index_result[1])}")

assert pd.isna(array_result[1]), f"Array should have NaN at position 1, got {array_result[1]}"
assert not pd.isna(index_result[1]), f"Index has NaN at position 1, expected {index_result[1]}"
print(f"\nBug confirmed: Index returns {index_result[1]} instead of NaN at position 1")