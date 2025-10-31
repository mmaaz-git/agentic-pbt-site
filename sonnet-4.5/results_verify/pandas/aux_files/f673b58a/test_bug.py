import io
from pandas.io.parsers import read_csv
import pandas as pd

print(f"Pandas version: {pd.__version__}")

# Test 1: Reproduce the exact bug report scenario
csv_content = "number\n1,000\n"
print("\n=== Test 1: Bug report scenario ===")
print(f"CSV content: {repr(csv_content)}")

df = read_csv(io.StringIO(csv_content), thousands=",")
print(f"Dataframe shape: {df.shape}")
print(f"Dataframe columns: {df.columns.tolist()}")
print(f"Dataframe:\n{df}")
print(f"First value (df.iloc[0]['number']): {df.iloc[0]['number']}")
print(f"Index: {df.index[0]}")

# Test 2: Multiple rows with thousands separator
csv_content2 = "number\n1,000\n2,500\n3,750\n"
print("\n=== Test 2: Multiple rows ===")
print(f"CSV content: {repr(csv_content2)}")

df2 = read_csv(io.StringIO(csv_content2), thousands=",")
print(f"Dataframe:\n{df2}")

# Test 3: Comparison with explicit sep different from thousands
csv_content3 = "number;value\n1,000;2,500\n3,750;4,250\n"
print("\n=== Test 3: Different sep (;) and thousands (,) ===")
print(f"CSV content: {repr(csv_content3)}")

df3 = read_csv(io.StringIO(csv_content3), sep=";", thousands=",")
print(f"Dataframe:\n{df3}")

# Test 4: Without thousands parameter
csv_content4 = "number\n1000\n2500\n3750\n"
print("\n=== Test 4: Without thousands separator ===")
print(f"CSV content: {repr(csv_content4)}")

df4 = read_csv(io.StringIO(csv_content4))
print(f"Dataframe:\n{df4}")

# Test 5: Explicit sep=',' and thousands=','
csv_content5 = "number\n1,000\n2,500\n"
print("\n=== Test 5: Explicit sep=',' and thousands=',' ===")
print(f"CSV content: {repr(csv_content5)}")

df5 = read_csv(io.StringIO(csv_content5), sep=",", thousands=",")
print(f"Dataframe:\n{df5}")
print(f"First row columns: {df5.columns.tolist()}")
if len(df5) > 0:
    print(f"First row values: {df5.iloc[0].tolist()}")