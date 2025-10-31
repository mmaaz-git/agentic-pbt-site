import pandas as pd
from io import StringIO

# Test empty DataFrame with different orient options
df = pd.DataFrame({"col_0": []})
print(f"Original DataFrame index: {df.index} (dtype: {df.index.dtype})\n")

orients = ['split', 'records', 'index', 'columns', 'values', 'table']

for orient in orients:
    try:
        json_str = df.to_json(orient=orient)
        df_roundtrip = pd.read_json(StringIO(json_str), orient=orient)

        print(f"Orient='{orient}':")
        print(f"  JSON: {json_str[:100]}...")
        print(f"  Roundtrip index: {df_roundtrip.index} (dtype: {df_roundtrip.index.dtype})")
        print(f"  Index dtype match: {df.index.dtype == df_roundtrip.index.dtype}")
        print()
    except Exception as e:
        print(f"Orient='{orient}': Failed - {e}")
        print()