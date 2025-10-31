import tempfile
import pandas as pd
import os

print("=" * 60)
print("Test Case 1: Single column with empty string")
print("=" * 60)
df = pd.DataFrame([{'text': ''}])
print("Original DataFrame:")
print(df)
print(f"Original shape: {df.shape}")
print(f"Original dtypes: {df.dtypes.to_dict()}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df.to_excel(tmp_path, index=False, engine='openpyxl')
    df_read = pd.read_excel(tmp_path, engine='openpyxl', na_filter=False)

    print("\nRead back DataFrame (with na_filter=False):")
    print(df_read)
    print(f"Read back shape: {df_read.shape}")
    print(f"Read back dtypes: {df_read.dtypes.to_dict()}")

    if len(df) != len(df_read):
        print(f"\n❌ ERROR: Row lost! Original had {len(df)} rows, read back has {len(df_read)} rows")
    else:
        print("\n✓ Row count preserved")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

print("\n" + "=" * 60)
print("Test Case 2: Single column with None value")
print("=" * 60)
df = pd.DataFrame([{'col': None}])
print("Original DataFrame:")
print(df)
print(f"Original shape: {df.shape}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df.to_excel(tmp_path, index=False, engine='openpyxl')
    df_read = pd.read_excel(tmp_path, engine='openpyxl')

    print("\nRead back DataFrame:")
    print(df_read)
    print(f"Read back shape: {df_read.shape}")

    if len(df) != len(df_read):
        print(f"\n❌ ERROR: Row lost! Original had {len(df)} rows, read back has {len(df_read)} rows")
    else:
        print("\n✓ Row count preserved")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

print("\n" + "=" * 60)
print("Test Case 3: Multiple columns, all empty/None")
print("=" * 60)
df = pd.DataFrame([{'col1': '', 'col2': None, 'col3': ''}])
print("Original DataFrame:")
print(df)
print(f"Original shape: {df.shape}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df.to_excel(tmp_path, index=False, engine='openpyxl')
    df_read = pd.read_excel(tmp_path, engine='openpyxl', na_filter=False)

    print("\nRead back DataFrame (with na_filter=False):")
    print(df_read)
    print(f"Read back shape: {df_read.shape}")

    if len(df) != len(df_read):
        print(f"\n❌ ERROR: Row lost! Original had {len(df)} rows, read back has {len(df_read)} rows")
    else:
        print("\n✓ Row count preserved")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

print("\n" + "=" * 60)
print("Test Case 4: Mixed case - one row all empty, one row with data")
print("=" * 60)
df = pd.DataFrame([{'col1': '', 'col2': None}, {'col1': 'data', 'col2': 42}])
print("Original DataFrame:")
print(df)
print(f"Original shape: {df.shape}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df.to_excel(tmp_path, index=False, engine='openpyxl')
    df_read = pd.read_excel(tmp_path, engine='openpyxl', na_filter=False)

    print("\nRead back DataFrame (with na_filter=False):")
    print(df_read)
    print(f"Read back shape: {df_read.shape}")

    if len(df) != len(df_read):
        print(f"\n❌ ERROR: Row lost! Original had {len(df)} rows, read back has {len(df_read)} rows")
    else:
        print("\n✓ Row count preserved")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

print("\n" + "=" * 60)
print("Test Case 5: From bug report - data=[(0, 0.0, '')]")
print("=" * 60)
data = [(0, 0.0, '')]
df = pd.DataFrame(data, columns=['int_col', 'float_col', 'str_col'])
print("Original DataFrame:")
print(df)
print(f"Original shape: {df.shape}")
print(f"Original values: {df.values.tolist()}")

with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    tmp_path = tmp.name

try:
    df.to_excel(tmp_path, index=False, engine='openpyxl')
    df_read = pd.read_excel(tmp_path, engine='openpyxl')

    print("\nRead back DataFrame:")
    print(df_read)
    print(f"Read back shape: {df_read.shape}")
    print(f"Read back values: {df_read.values.tolist()}")

    if len(df) != len(df_read):
        print(f"\n❌ ERROR: Row lost! Original had {len(df)} rows, read back has {len(df_read)} rows")
    else:
        print("\n✓ Row count preserved")

    # Check value preservation
    if len(df_read) > 0:
        if df_read['str_col'].isna().iloc[0]:
            print("❌ ERROR: Empty string converted to NaN")
        else:
            print("✓ Empty string preserved")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)