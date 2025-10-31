import pandas as pd
from io import StringIO

# Test with the specific failing value mentioned
print("Testing with specific value: 1.5932223682757467")
value = 1.5932223682757467
df = pd.DataFrame({'col': [value]})
json_str = df.to_json(orient='records')
df_restored = pd.read_json(StringIO(json_str), orient='records')

orig = df['col'].iloc[0]
restored = df_restored['col'].iloc[0]

print(f"Original:  {orig:.17f}")
print(f"JSON:      {json_str}")
print(f"Restored:  {restored:.17f}")
print(f"Match:     {orig == restored}")

if orig != restored:
    print(f"Test failed: Round-trip failed: {orig} != {restored}")
else:
    print("Test passed!")

# Test the second example from the bug report
print("\n\nTesting with the full example from the bug report:")
df = pd.DataFrame({'values': [1.5932223682757467, 0.0013606744423365084]})
print(f"Original:  {df['values'].iloc[0]:.17f}")

json_str = df.to_json(orient='records')
print(f"JSON:      {json_str}")

df_restored = pd.read_json(StringIO(json_str), orient='records')
print(f"Restored:  {df_restored['values'].iloc[0]:.17f}")
print(f"Match:     {df['values'].iloc[0] == df_restored['values'].iloc[0]}")

# Test with double_precision=15 to see if it fixes the issue
print("\n\nTesting with double_precision=15:")
df = pd.DataFrame({'values': [1.5932223682757467, 0.0013606744423365084]})
print(f"Original:  {df['values'].iloc[0]:.17f}")

json_str = df.to_json(orient='records', double_precision=15)
print(f"JSON:      {json_str}")

df_restored = pd.read_json(StringIO(json_str), orient='records')
print(f"Restored:  {df_restored['values'].iloc[0]:.17f}")
print(f"Match:     {df['values'].iloc[0] == df_restored['values'].iloc[0]}")