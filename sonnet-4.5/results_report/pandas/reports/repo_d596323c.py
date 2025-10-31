import pandas as pd
from io import StringIO

# Create a DataFrame with all-numeric string column names
df = pd.DataFrame({'00': [1, 2], '0': [3, 4]})
print("Original DataFrame:")
print(df)
print("\nOriginal column names:", list(df.columns))
print("Original column types:", [type(c) for c in df.columns])

# Convert to JSON
json_str = df.to_json(orient='split')
print("\nJSON representation:")
print(json_str)

# Read back from JSON
result = pd.read_json(StringIO(json_str), orient='split')
print("\nResulting DataFrame after round-trip:")
print(result)
print("\nResult column names:", list(result.columns))
print("Result column types:", [type(c) for c in result.columns])

# Check for data loss
print("\nData loss detected:")
print(f"  '00' became: {result.columns[0]} (type: {type(result.columns[0])})")
print(f"  '0' became: {result.columns[1]} (type: {type(result.columns[1])})")

# Show that both columns became the same value
if result.columns[0] == result.columns[1]:
    print(f"\nCRITICAL: Both columns '00' and '0' became the same value: {result.columns[0]}")