import pandas as pd
import io

df = pd.DataFrame([['']],  columns=['col0'])
csv_str = df.to_csv(index=False)
df_roundtrip = pd.read_csv(io.StringIO(csv_str))

print("Original value:", repr(df.iloc[0, 0]))
print("After round-trip:", repr(df_roundtrip.iloc[0, 0]))
print("Round-trip preserves data:", df.equals(df_roundtrip))

print("\nCSV output:")
print(csv_str)