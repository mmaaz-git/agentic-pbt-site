import pandas as pd
import io

df = pd.DataFrame({'col_0': [0]})
df.index = df.index.astype(str)

print("Original index:", list(df.index), "- dtype:", df.index.dtype)

json_str = df.to_json(orient='index')
print("JSON string:", json_str)
df_roundtrip = pd.read_json(io.StringIO(json_str), orient='index')

print("Roundtrip index:", list(df_roundtrip.index), "- dtype:", df_roundtrip.index.dtype)
print("Equal?", df.equals(df_roundtrip))

# Test with columns as well
print("\n--- Testing with columns ---")
df2 = pd.DataFrame({str(i): [i] for i in range(3)})
print("Original columns:", list(df2.columns), "- dtype:", df2.columns.dtype)

json_str2 = df2.to_json(orient='columns')
print("JSON string:", json_str2)
df2_roundtrip = pd.read_json(io.StringIO(json_str2), orient='columns')

print("Roundtrip columns:", list(df2_roundtrip.columns), "- dtype:", df2_roundtrip.columns.dtype)
print("Equal?", df2.equals(df2_roundtrip))

# Test mixed indices (non-numeric and numeric strings)
print("\n--- Testing mixed indices ---")
df3 = pd.DataFrame({'col_0': [0, 1, 2]})
df3.index = ['0', 'a', '2']
print("Original mixed index:", list(df3.index), "- dtype:", df3.index.dtype)

json_str3 = df3.to_json(orient='index')
print("JSON string:", json_str3)
df3_roundtrip = pd.read_json(io.StringIO(json_str3), orient='index')

print("Roundtrip mixed index:", list(df3_roundtrip.index), "- dtype:", df3_roundtrip.index.dtype)
print("Equal?", df3.equals(df3_roundtrip))

# Test workaround with convert_axes=False
print("\n--- Testing workaround with convert_axes=False ---")
df4 = pd.DataFrame({'col_0': [0]})
df4.index = df4.index.astype(str)

print("Original index:", list(df4.index), "- dtype:", df4.index.dtype)

json_str4 = df4.to_json(orient='index')
df4_roundtrip = pd.read_json(io.StringIO(json_str4), orient='index', convert_axes=False)

print("Roundtrip index (convert_axes=False):", list(df4_roundtrip.index), "- dtype:", df4_roundtrip.index.dtype)
print("Equal?", df4.equals(df4_roundtrip))