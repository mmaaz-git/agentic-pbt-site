import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe

# Create a DataFrame with nullable boolean dtype containing NA values
df = pd.DataFrame({'col': pd.array([True, False, None], dtype='boolean')})
print("Original DataFrame:")
print(df)
print(f"DataFrame dtype: {df['col'].dtype}")
print(f"Has NA values? {df['col'].isna().any()}")
print(f"NA count: {df['col'].isna().sum()}")

# Convert to interchange format and back
print("\n--- Converting through interchange protocol ---")
interchange_obj = df.__dataframe__()
result = from_dataframe(interchange_obj)

print("\nAfter round-trip through interchange:")
print(result)
print(f"DataFrame dtype: {result['col'].dtype}")
print(f"Has NA values? {result['col'].isna().any()}")
print(f"NA count: {result['col'].isna().sum()}")

# Show individual values
print("\n--- Comparing individual values ---")
print("Original values:")
for i, val in enumerate(df['col']):
    print(f"  Index {i}: {val}")

print("\nRound-trip values:")
for i, val in enumerate(result['col']):
    print(f"  Index {i}: {val}")