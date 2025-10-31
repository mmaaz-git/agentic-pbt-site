import pandas as pd
import pandas.util
import numpy as np

print("Testing hash_pandas_object with short hash_key:")
series = pd.Series([1, 2, 3])

try:
    result = pandas.util.hash_pandas_object(series, hash_key="test")
    print(f"Result: {result}")
except ValueError as e:
    print(f"Error: {e}")

print("\nTesting hash_array with short hash_key:")
arr = np.array([1, 2, 3])

try:
    result = pandas.util.hash_array(arr, hash_key="test")
    print(f"Result: {result}")
except ValueError as e:
    print(f"Error: {e}")

print("\nTesting with 16-byte key:")
try:
    result = pandas.util.hash_pandas_object(series, hash_key="0123456789abcdef")
    print(f"Result (16-byte key): {result}")
except ValueError as e:
    print(f"Error: {e}")