import pandas as pd
import io

# Test specifically with tab character
name = '\t'
df = pd.DataFrame([[1]], columns=[name])
csv_str = df.to_csv(index=False)
result = pd.read_csv(io.StringIO(csv_str))

print(f"Original column name: {repr(name)}")
print(f"Result column name: {repr(result.columns[0])}")
print(f"Column names match: {result.columns[0] == name}")
print(f"Result values: {result.values.tolist()}")
print()

# Also test null character
name = '\x00'
df = pd.DataFrame([[1]], columns=[name])
csv_str = df.to_csv(index=False)
result = pd.read_csv(io.StringIO(csv_str))

print(f"Original column name (null): {repr(name)}")
print(f"Result column name: {repr(result.columns[0])}")
print(f"Column names match: {result.columns[0] == name}")
print(f"Result values: {result.values.tolist()}")