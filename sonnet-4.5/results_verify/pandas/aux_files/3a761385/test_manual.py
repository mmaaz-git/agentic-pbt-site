import tempfile
import pandas as pd
import os

df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'status': ['active', '']
})

print("Original DataFrame:")
print(df)
print(f"DataFrame dtypes: {df.dtypes.tolist()}")
print(f"Status column type: {type(df['status'][1])}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
    filepath = f.name

df.to_excel(filepath, index=False)
df_read = pd.read_excel(filepath)

print("\nAfter round-trip:")
print(df_read)
print(f"DataFrame dtypes: {df_read.dtypes.tolist()}")
print(f"Status column type for Bob: {type(df_read['status'][1])}")

print("\nData loss:")
print(f"Original: {df['status'].tolist()}")
print(f"After: {df_read['status'].tolist()}")

# Test with different parameters
print("\n" + "="*50)
print("Testing with keep_default_na=False:")
df_read2 = pd.read_excel(filepath, keep_default_na=False)
print(df_read2)
print(f"After with keep_default_na=False: {df_read2['status'].tolist()}")

print("\n" + "="*50)
print("Testing with na_filter=False:")
df_read3 = pd.read_excel(filepath, na_filter=False)
print(df_read3)
print(f"After with na_filter=False: {df_read3['status'].tolist()}")

# Clean up
os.unlink(filepath)