import pandas as pd
import io

print("=" * 60)
print("Testing DataFrame with 1 row and 0 columns")
print("=" * 60)

df = pd.DataFrame([{}])
print(f'Original DataFrame shape: {df.shape}')
print(f'Original DataFrame:\n{df}')
print(f'Original DataFrame length: {len(df)}')

print("\nTesting different orient values:")

# Test orient='records'
print("\n--- orient='records' ---")
json_str = df.to_json(orient='records')
print(f'JSON output: {json_str}')
df_back = pd.read_json(io.StringIO(json_str), orient='records')
print(f'Restored DataFrame shape: {df_back.shape}')
print(f'Restored DataFrame:\n{df_back}')
print(f'Original length: {len(df)}, Restored length: {len(df_back)}')
print(f'Roundtrip successful: {len(df) == len(df_back)}')

# Test orient='index'
print("\n--- orient='index' ---")
json_str = df.to_json(orient='index')
print(f'JSON output: {json_str}')
df_back = pd.read_json(io.StringIO(json_str), orient='index')
print(f'Restored DataFrame shape: {df_back.shape}')
print(f'Restored DataFrame:\n{df_back}')
print(f'Original length: {len(df)}, Restored length: {len(df_back)}')
print(f'Roundtrip successful: {len(df) == len(df_back)}')

# Test orient='columns'
print("\n--- orient='columns' ---")
json_str = df.to_json(orient='columns')
print(f'JSON output: {json_str}')
df_back = pd.read_json(io.StringIO(json_str), orient='columns')
print(f'Restored DataFrame shape: {df_back.shape}')
print(f'Restored DataFrame:\n{df_back}')
print(f'Original length: {len(df)}, Restored length: {len(df_back)}')
print(f'Roundtrip successful: {len(df) == len(df_back)}')

# Test orient='values'
print("\n--- orient='values' ---")
json_str = df.to_json(orient='values')
print(f'JSON output: {json_str}')
df_back = pd.read_json(io.StringIO(json_str), orient='values')
print(f'Restored DataFrame shape: {df_back.shape}')
print(f'Restored DataFrame:\n{df_back}')
print(f'Original length: {len(df)}, Restored length: {len(df_back)}')
print(f'Roundtrip successful: {len(df) == len(df_back)}')

# Test orient='split'
print("\n--- orient='split' ---")
json_str = df.to_json(orient='split')
print(f'JSON output: {json_str}')
df_back = pd.read_json(io.StringIO(json_str), orient='split')
print(f'Restored DataFrame shape: {df_back.shape}')
print(f'Restored DataFrame:\n{df_back}')
print(f'Original length: {len(df)}, Restored length: {len(df_back)}')
print(f'Roundtrip successful: {len(df) == len(df_back)}')

# Test orient='table'
print("\n--- orient='table' ---")
json_str = df.to_json(orient='table')
print(f'JSON output: {json_str}')
df_back = pd.read_json(io.StringIO(json_str), orient='table')
print(f'Restored DataFrame shape: {df_back.shape}')
print(f'Restored DataFrame:\n{df_back}')
print(f'Original length: {len(df)}, Restored length: {len(df_back)}')
print(f'Roundtrip successful: {len(df) == len(df_back)}')

# Test with multiple rows
print("\n" + "=" * 60)
print("Testing DataFrame with 3 rows and 0 columns")
print("=" * 60)

df2 = pd.DataFrame([{}, {}, {}])
print(f'\nOriginal DataFrame shape: {df2.shape}')
print(f'Original DataFrame:\n{df2}')

json_str2 = df2.to_json(orient='records')
print(f'\nJSON output (orient="records"): {json_str2}')
df2_back = pd.read_json(io.StringIO(json_str2), orient='records')
print(f'Restored DataFrame shape: {df2_back.shape}')
print(f'Original length: {len(df2)}, Restored length: {len(df2_back)}')
print(f'Roundtrip successful: {len(df2) == len(df2_back)}')