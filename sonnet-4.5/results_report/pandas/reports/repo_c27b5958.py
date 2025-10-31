import io
import pandas as pd

# Create an empty DataFrame with integer and float columns
df = pd.DataFrame({"a": [], "b": []})
df["a"] = df["a"].astype(int)
df["b"] = df["b"].astype(float)

print("=== Empty DataFrame with orient='split' ===")
print(f"Original index dtype: {df.index.dtype}")
print(f"Original DataFrame:\n{df}")
print(f"Original DataFrame.dtypes:\n{df.dtypes}")

# Convert to JSON with orient='split'
json_str = df.to_json(orient="split")
print(f"\nJSON representation: {json_str}")

# Read back from JSON
result = pd.read_json(io.StringIO(json_str), orient="split")
print(f"\nResult index dtype: {result.index.dtype}")
print(f"Result DataFrame:\n{result}")
print(f"Result DataFrame.dtypes:\n{result.dtypes}")

# Check if dtypes match
print(f"\nIndex dtype matches: {df.index.dtype == result.index.dtype}")

print("\n=== Testing with orient='columns' ===")
json_str_columns = df.to_json(orient="columns")
print(f"JSON representation: {json_str_columns}")
result_columns = pd.read_json(io.StringIO(json_str_columns), orient="columns")
print(f"Result index dtype: {result_columns.index.dtype}")
print(f"Index dtype matches: {df.index.dtype == result_columns.index.dtype}")

print("\n=== Testing with non-empty DataFrame (for comparison) ===")
df_nonempty = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
print(f"Original non-empty index dtype: {df_nonempty.index.dtype}")
json_str_nonempty = df_nonempty.to_json(orient="split")
result_nonempty = pd.read_json(io.StringIO(json_str_nonempty), orient="split")
print(f"Result non-empty index dtype: {result_nonempty.index.dtype}")
print(f"Index dtype matches: {df_nonempty.index.dtype == result_nonempty.index.dtype}")