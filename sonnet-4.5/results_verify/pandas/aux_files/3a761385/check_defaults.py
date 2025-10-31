import pandas as pd

# Check default NA values using pandas.io.common
try:
    from pandas.io.common import STR_NA_VALUES
    print("Default NA values in pandas:")
    print(STR_NA_VALUES)
except:
    print("Could not import STR_NA_VALUES directly")

# Also test what happens with None
import tempfile

df = pd.DataFrame({
    'empty_string': [''],
    'none_value': [None],
    'nan_value': [float('nan')]
})

print("\nOriginal DataFrame:")
print(df)
print(f"empty_string type: {type(df['empty_string'][0])}")
print(f"none_value type: {type(df['none_value'][0])}")
print(f"nan_value type: {type(df['nan_value'][0])}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
    filepath = f.name

df.to_excel(filepath, index=False)

# Read back with default settings
df_default = pd.read_excel(filepath)
print("\nAfter round-trip (default settings):")
print(df_default)
print(f"empty_string: {df_default['empty_string'].tolist()}")
print(f"none_value: {df_default['none_value'].tolist()}")
print(f"nan_value: {df_default['nan_value'].tolist()}")

# Read back with na_filter=False
df_no_filter = pd.read_excel(filepath, na_filter=False)
print("\nAfter round-trip (na_filter=False):")
print(df_no_filter)
print(f"empty_string: {df_no_filter['empty_string'].tolist()}")
print(f"none_value: {df_no_filter['none_value'].tolist()}")
print(f"nan_value: {df_no_filter['nan_value'].tolist()}")

import os
os.unlink(filepath)