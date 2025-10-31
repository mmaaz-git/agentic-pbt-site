import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe

# Test 1: Multiple NA values
df1 = pd.DataFrame({'col': pd.array([None, None, True, None], dtype='boolean')})
print("Test 1 - Multiple NAs:")
print("Original:", df1['col'].tolist())
result1 = from_dataframe(df1.__dataframe__())
print("After round-trip:", result1['col'].tolist())

# Test 2: All NA values
df2 = pd.DataFrame({'col': pd.array([None, None, None], dtype='boolean')})
print("\nTest 2 - All NAs:")
print("Original:", df2['col'].tolist())
result2 = from_dataframe(df2.__dataframe__())
print("After round-trip:", result2['col'].tolist())

# Test 3: No NA values
df3 = pd.DataFrame({'col': pd.array([True, False, True], dtype='boolean')})
print("\nTest 3 - No NAs:")
print("Original:", df3['col'].tolist())
result3 = from_dataframe(df3.__dataframe__())
print("After round-trip:", result3['col'].tolist())
print("Dtypes match?", df3['col'].dtype == result3['col'].dtype)