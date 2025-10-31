import pandas as pd
import tempfile
import os

# Test with mixed content
df = pd.DataFrame({
    'col1': ['a', '', 'c'],
    'col2': ['', 'b', ''],
    'col3': ['', '', '']
})

print("Original DataFrame:")
print(df)
print(f"Shape: {df.shape}")
print(f"Values: {df.values.tolist()}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
    filepath = f.name

df.to_excel(filepath, index=False)

# Read back with default settings
df_default = pd.read_excel(filepath)
print("\nAfter round-trip (default settings):")
print(df_default)
print(f"Shape: {df_default.shape}")
print(f"Values: {df_default.values.tolist()}")

# Read back with na_filter=False
df_no_filter = pd.read_excel(filepath, na_filter=False)
print("\nAfter round-trip (na_filter=False):")
print(df_no_filter)
print(f"Shape: {df_no_filter.shape}")
print(f"Values: {df_no_filter.values.tolist()}")

# Test a row with only empty strings
df2 = pd.DataFrame([['', '', '']], columns=['A', 'B', 'C'])
print("\n" + "="*50)
print("Testing DataFrame with only empty strings:")
print(df2)

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
    filepath2 = f.name

df2.to_excel(filepath2, index=False)
df2_read = pd.read_excel(filepath2)
print("\nAfter round-trip (default):")
print(df2_read)
print(f"Shape: Original={df2.shape}, After={df2_read.shape}")

df2_read_no_filter = pd.read_excel(filepath2, na_filter=False)
print("\nAfter round-trip (na_filter=False):")
print(df2_read_no_filter)
print(f"Shape: Original={df2.shape}, After={df2_read_no_filter.shape}")

os.unlink(filepath)
os.unlink(filepath2)