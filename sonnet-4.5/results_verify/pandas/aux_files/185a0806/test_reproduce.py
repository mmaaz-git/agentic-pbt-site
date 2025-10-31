import pandas as pd
import numpy as np
from pandas.api.extensions import take

arr = np.array([10.0, 20.0, 30.0])
series = pd.Series([10.0, 20.0, 30.0])
index = pd.Index([10, 20, 30])

print("numpy array with allow_fill=True:")
result = take(arr, [0, -1], allow_fill=True)
print(f"  {result}")

print("\nSeries with allow_fill=True:")
try:
    result = take(series, [0, -1], allow_fill=True)
    print(f"  {result}")
except TypeError as e:
    print(f"  TypeError: {e}")

print("\nIndex with allow_fill=True, fill_value=None:")
result = take(index, [0, -1], allow_fill=True, fill_value=None)
print(f"  {result}")
print(f"  Second element: {result[1]} (expected NaN, got last element)")

print("\nIndex with allow_fill=True, fill_value=-999:")
try:
    result = take(index, [0, -1], allow_fill=True, fill_value=-999)
    print(f"  {result}")
except ValueError as e:
    print(f"  ValueError: {e}")