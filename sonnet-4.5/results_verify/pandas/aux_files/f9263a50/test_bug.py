import pandas as pd
from io import StringIO

# Test case from bug report
df = pd.DataFrame({"col_0": []})

print(f"Original DataFrame:")
print(df)
print(f"Original index type: {df.index.inferred_type}")
print(f"Original index dtype: {df.index.dtype}")
print(f"Original index: {df.index}")
print()

# Convert to JSON with orient='split'
json_str = df.to_json(orient='split')
print(f"JSON representation: {json_str}")
print()

# Read back from JSON
df_roundtrip = pd.read_json(StringIO(json_str), orient='split')

print(f"Roundtrip DataFrame:")
print(df_roundtrip)
print(f"Roundtrip index type: {df_roundtrip.index.inferred_type}")
print(f"Roundtrip index dtype: {df_roundtrip.index.dtype}")
print(f"Roundtrip index: {df_roundtrip.index}")
print()

print(f"Bug claim: {df.index.inferred_type} != {df_roundtrip.index.inferred_type}")
print(f"Index types match: {df.index.inferred_type == df_roundtrip.index.inferred_type}")
print(f"Index dtypes match: {df.index.dtype == df_roundtrip.index.dtype}")