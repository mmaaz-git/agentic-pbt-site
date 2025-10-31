import pandas as pd
from io import StringIO

# Test with the specific failing value from the bug report
df = pd.DataFrame({'col': [1.5932223682757467]})
print(f"Original value:  {df['col'].iloc[0]:.17f}")

# Serialize to JSON with default settings
json_str = df.to_json(orient='records')
print(f"JSON output:     {json_str}")

# Restore from JSON
df_restored = pd.read_json(StringIO(json_str), orient='records')
print(f"Restored value:  {df_restored['col'].iloc[0]:.17f}")

# Check equality
print(f"Values equal:    {df['col'].iloc[0] == df_restored['col'].iloc[0]}")

# Show the precision loss
print(f"Difference:      {abs(df['col'].iloc[0] - df_restored['col'].iloc[0]):.20e}")

print("\n--- Testing with double_precision=15 ---")

# Test with increased precision
json_str_15 = df.to_json(orient='records', double_precision=15)
print(f"JSON output:     {json_str_15}")

df_restored_15 = pd.read_json(StringIO(json_str_15), orient='records')
print(f"Restored value:  {df_restored_15['col'].iloc[0]:.17f}")
print(f"Values equal:    {df['col'].iloc[0] == df_restored_15['col'].iloc[0]}")
print(f"Difference:      {abs(df['col'].iloc[0] - df_restored_15['col'].iloc[0]):.20e}")

print("\n--- Testing with Python's standard json module ---")
import json

# Using Python's standard JSON library for comparison
data = {'col': [1.5932223682757467]}
json_str_std = json.dumps(data)
print(f"JSON output:     {json_str_std}")

data_restored = json.loads(json_str_std)
print(f"Restored value:  {data_restored['col'][0]:.17f}")
print(f"Values equal:    {data['col'][0] == data_restored['col'][0]}")