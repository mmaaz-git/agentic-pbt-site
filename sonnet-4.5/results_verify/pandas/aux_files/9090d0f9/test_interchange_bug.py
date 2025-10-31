import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe

print("=== Test 1: Basic reproduction ===")
df = pd.DataFrame({'col': pd.array([1, 2, None], dtype='Int64')})
print("Original DataFrame:")
print(df)
print(f"Original dtype: {df['col'].dtype}")
print(f"Original values: {df['col'].values}")

interchange_obj = df.__dataframe__()
result = from_dataframe(interchange_obj)
print("\nAfter round-trip through interchange:")
print(result)
print(f"Result dtype: {result['col'].dtype}")
print(f"Result values: {result['col'].values}")

print("\n=== Test 2: Without NA values ===")
df_no_na = pd.DataFrame({'col': pd.array([1, 2, 3], dtype='Int64')})
print("Original DataFrame (no NA):")
print(df_no_na)
print(f"Original dtype: {df_no_na['col'].dtype}")

result_no_na = from_dataframe(df_no_na.__dataframe__())
print("\nAfter round-trip (no NA):")
print(result_no_na)
print(f"Result dtype: {result_no_na['col'].dtype}")

print("\n=== Test 3: Multiple columns with mixed types ===")
df_mixed = pd.DataFrame({
    'int_with_na': pd.array([1, None, 3], dtype='Int64'),
    'int_no_na': pd.array([4, 5, 6], dtype='Int64'),
    'float': [1.1, 2.2, 3.3],
    'string': ['a', 'b', 'c']
})
print("Original mixed DataFrame:")
print(df_mixed)
print("\nOriginal dtypes:")
for col in df_mixed.columns:
    print(f"  {col}: {df_mixed[col].dtype}")

result_mixed = from_dataframe(df_mixed.__dataframe__())
print("\nAfter round-trip mixed DataFrame:")
print(result_mixed)
print("\nResult dtypes:")
for col in result_mixed.columns:
    print(f"  {col}: {result_mixed[col].dtype}")

print("\n=== Test 4: Check equality ===")
try:
    pd.testing.assert_frame_equal(df, result)
    print("DataFrames are equal")
except AssertionError as e:
    print(f"DataFrames are NOT equal: {e}")