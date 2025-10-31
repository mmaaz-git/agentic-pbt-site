import pandas as pd
import pandas.util
import numpy as np

print("Testing with specific value from hypothesis test:")
series = pd.Series([-9_223_372_036_854_775_809])

print("\nWith 'test' as hash_key:")
try:
    result = pandas.util.hash_pandas_object(series, hash_key="test")
    print(f"Result: {result}")
except ValueError as e:
    print(f"Error: {e}")

print("\nWith 'key1' as hash_key:")
try:
    result = pandas.util.hash_pandas_object(series, hash_key="key1")
    print(f"Result: {result}")
except ValueError as e:
    print(f"Error: {e}")

print("\nWith default hash_key:")
try:
    result = pandas.util.hash_pandas_object(series)
    print(f"Result: {result}")
except ValueError as e:
    print(f"Error: {e}")

print("\nWith 16-byte key:")
try:
    result = pandas.util.hash_pandas_object(series, hash_key="0123456789abcdef")
    print(f"Result: {result}")
except ValueError as e:
    print(f"Error: {e}")