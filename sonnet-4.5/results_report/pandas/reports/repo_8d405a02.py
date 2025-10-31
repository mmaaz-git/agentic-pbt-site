import pandas as pd
import io

df = pd.DataFrame([[42]], columns=['\t'])
print("Original:")
print(f"  Columns: {list(df.columns)}")
print(f"  Values: {df.values.tolist()}")

csv_str = df.to_csv(index=False)
print("\nCSV output:")
print(repr(csv_str))

result = pd.read_csv(io.StringIO(csv_str))

print("\nAfter round-trip:")
print(f"  Columns: {list(result.columns)}")
print(f"  Values: {result.values.tolist()}")