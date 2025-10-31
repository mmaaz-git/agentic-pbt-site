import pandas as pd
import numpy as np
from pandas.io.json._table_schema import build_table_schema, parse_table_schema
import json

# Test different integer types
data = {
    'int8': np.int8(127),
    'int16': np.int16(32767),
    'int32': np.int32(2147483647),
    'int64': np.int64(9223372036854775807),
    'uint8': np.uint8(255),
    'uint16': np.uint16(65535),
    'uint32': np.uint32(4294967295),
    'uint64': np.uint64(9223372036854775808),  # 2^63
}

df = pd.DataFrame([data])
print("Original DataFrame dtypes:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

schema = build_table_schema(df, index=False)
print("\nGenerated Schema:")
for field in schema['fields']:
    print(f"  {field['name']}: {field['type']}")

# Now test round-trip with table orient
json_str = df.to_json(orient='table')
json_data = json.loads(json_str)
print("\nJSON data values:")
for key, val in json_data['data'][0].items():
    if key != 'index':
        print(f"  {key}: {val} (type: {type(val).__name__})")

# Parse back
df_roundtrip = pd.read_json(json_str, orient='table')
print("\nRoundtrip DataFrame dtypes:")
for col in df_roundtrip.columns:
    print(f"  {col}: {df_roundtrip[col].dtype}")

print("\nValue comparison:")
for col in df.columns:
    orig = df[col].iloc[0]
    rt = df_roundtrip[col].iloc[0]
    print(f"  {col}: {orig} -> {rt} (match: {orig == rt})")