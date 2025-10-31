import pandas as pd
from io import StringIO

# Test with value just outside int64 range
df = pd.DataFrame([{'col': -9223372036854775809}])
print(f"Original DataFrame:\n{df}")
print(f"Data type: {df['col'].dtype}")

json_str = df.to_json(orient='split')
print(f"\nJSON string created successfully:\n{json_str}")

try:
    df_back = pd.read_json(StringIO(json_str), orient='split')
    print(f"\nRound-trip successful:\n{df_back}")
except ValueError as e:
    print(f"\nError during read_json: {e}")

# Test with value within int64 range
print("\n" + "="*50)
print("Testing with value within int64 range:")
df2 = pd.DataFrame([{'col': -9223372036854775808}])  # Minimum int64
print(f"Original DataFrame:\n{df2}")

json_str2 = df2.to_json(orient='split')
print(f"\nJSON string:\n{json_str2}")

try:
    df_back2 = pd.read_json(StringIO(json_str2), orient='split')
    print(f"\nRound-trip successful:\n{df_back2}")
except ValueError as e:
    print(f"\nError during read_json: {e}")