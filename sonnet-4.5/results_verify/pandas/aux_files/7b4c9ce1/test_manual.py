import tempfile
import pandas as pd
import os

print("=" * 60)
print("Test 1: DataFrame with empty string")
print("=" * 60)

df = pd.DataFrame([{'text': ''}])
print("Original:", df)
print("Length:", len(df))
print("Values:", df.values)
print("Type of value:", type(df.iloc[0, 0]))

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

df.to_excel(tmp_path, index=False, engine='openpyxl')
df_read = pd.read_excel(tmp_path, engine='openpyxl', na_filter=False)

print("\nRead back:", df_read)
print("Length:", len(df_read))
if len(df_read) > 0:
    print("Values:", df_read.values)
    print("Type of value:", type(df_read.iloc[0, 0]) if len(df_read) > 0 else "No data")

os.unlink(tmp_path)

print("\n" + "=" * 60)
print("Test 2: DataFrame with None value")
print("=" * 60)

df = pd.DataFrame([{'col': None}])
print("Original:", df)
print("Length:", len(df))
print("Values:", df.values)

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name
df.to_excel(tmp_path, index=False, engine='openpyxl')
df_read = pd.read_excel(tmp_path, engine='openpyxl')
print("\nRead back:", df_read)
print(f"Original: {len(df)} rows, Read back: {len(df_read)} rows")

os.unlink(tmp_path)

print("\n" + "=" * 60)
print("Test 3: DataFrame with multiple columns, all empty/None")
print("=" * 60)

df = pd.DataFrame([{'col1': '', 'col2': None, 'col3': ''}])
print("Original:", df)
print("Length:", len(df))

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name
df.to_excel(tmp_path, index=False, engine='openpyxl')
df_read = pd.read_excel(tmp_path, engine='openpyxl')
print("\nRead back:", df_read)
print(f"Original: {len(df)} rows, Read back: {len(df_read)} rows")

os.unlink(tmp_path)

print("\n" + "=" * 60)
print("Test 4: DataFrame with mixed data - one empty row, one with data")
print("=" * 60)

df = pd.DataFrame([{'col1': '', 'col2': None}, {'col1': 'data', 'col2': 'more'}])
print("Original:", df)
print("Length:", len(df))

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name
df.to_excel(tmp_path, index=False, engine='openpyxl')
df_read = pd.read_excel(tmp_path, engine='openpyxl', na_filter=False)
print("\nRead back:", df_read)
print(f"Original: {len(df)} rows, Read back: {len(df_read)} rows")

os.unlink(tmp_path)