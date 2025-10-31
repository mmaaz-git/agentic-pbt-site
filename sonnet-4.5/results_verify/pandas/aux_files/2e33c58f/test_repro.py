import pandas as pd
import io

df = pd.DataFrame({'col': [9223372036854775808]})

print(f"Original value: {df['col'][0]}")
print(f"Original dtype: {df['col'].dtype}")

json_str = df.to_json(orient='table')
print(f"\nJSON string produced:\n{json_str}\n")

df_roundtrip = pd.read_json(io.StringIO(json_str), orient='table')

print(f"Roundtrip value: {df_roundtrip['col'][0]}")
print(f"Roundtrip dtype: {df_roundtrip['col'].dtype}")
print(f"Data corrupted: {df['col'][0] != df_roundtrip['col'][0]}")

# Test with other orients for comparison
print("\n--- Testing with other orients ---")
for orient in ['split', 'records', 'index', 'columns', 'values']:
    try:
        json_str_alt = df.to_json(orient=orient)
        df_rt_alt = pd.read_json(io.StringIO(json_str_alt), orient=orient)
        print(f"{orient}: original={df['col'][0]}, roundtrip={df_rt_alt['col'][0] if 'col' in df_rt_alt else df_rt_alt.iloc[0,0]}, match={df['col'][0] == (df_rt_alt['col'][0] if 'col' in df_rt_alt else df_rt_alt.iloc[0,0])}")
    except Exception as e:
        print(f"{orient}: Error - {e}")