import pandas as pd
import numpy as np

categories = ['a', 'b', 'c']
codes = [0, -1, 2]
df = pd.DataFrame({'cat': pd.Categorical.from_codes(codes, categories=categories)})

print("Original DataFrame:")
print(df)
print(f"Original values: {list(df['cat'])}")
print(f"Null values: {df['cat'].isna().sum()}")

interchange_obj = df.__dataframe__()
result = pd.api.interchange.from_dataframe(interchange_obj)

print("\nResult after round-trip:")
print(result)
print(f"Result values: {list(result['cat'])}")
print(f"Null values: {result['cat'].isna().sum()}")

print("\nAssertion check:")
print(f"Expected: {list(df['cat'])}")
print(f"Got: {list(result['cat'])}")
assert list(df['cat']) == list(result['cat']), f"Values don't match!"