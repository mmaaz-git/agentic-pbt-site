import pandas as pd
from io import StringIO

# Test case that demonstrates the bug
df = pd.DataFrame([{'0': 123}])
print(f"Original columns: {df.columns.tolist()}")
print(f"Column type: {type(df.columns[0])}")

json_str = df.to_json(orient='records')
print(f"JSON: {json_str}")

df_back = pd.read_json(StringIO(json_str), orient='records')
print(f"Round-trip columns: {df_back.columns.tolist()}")
print(f"Column type: {type(df_back.columns[0])}")
print(f"Columns equal? {df.columns.equals(df_back.columns)}")

# Demonstrate the KeyError that results
print("\nTrying to access column by original name:")
try:
    print(f"df['0'] = {df['0'].tolist()}")
except KeyError as e:
    print(f"KeyError on original df: {e}")

try:
    print(f"df_back['0'] = {df_back['0'].tolist()}")
except KeyError as e:
    print(f"KeyError on round-trip df: {e}")

# Show how to access it after round-trip
print(f"df_back[0] = {df_back[0].tolist()}")