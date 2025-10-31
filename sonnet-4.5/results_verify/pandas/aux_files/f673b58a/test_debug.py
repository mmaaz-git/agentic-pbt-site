import io
from pandas.io.parsers import read_csv
import pandas as pd

# Let's understand what's happening step by step
csv_content = "number\n1,000\n2,500\n"
print("CSV content:")
print(csv_content)
print()

# Test 1: Default behavior (sep=',', no thousands)
print("=== Test 1: Default (sep=',', no thousands) ===")
df1 = read_csv(io.StringIO(csv_content))
print(f"Shape: {df1.shape}")
print(f"Columns: {df1.columns.tolist()}")
print("DataFrame:")
print(df1)
print()

# Test 2: With thousands=',' (implicitly sep=',')
print("=== Test 2: thousands=',' (implicitly sep=',') ===")
df2 = read_csv(io.StringIO(csv_content), thousands=",")
print(f"Shape: {df2.shape}")
print(f"Columns: {df2.columns.tolist()}")
print("DataFrame:")
print(df2)
print()

# Test 3: Try different CSV format with semicolon separator
csv_semicolon = "number;value\n1,000;2,500\n3,750;4,250\n"
print("=== Test 3: CSV with semicolon separator ===")
print("CSV content:")
print(csv_semicolon)
df3 = read_csv(io.StringIO(csv_semicolon), sep=";", thousands=",")
print(f"Shape: {df3.shape}")
print(f"Columns: {df3.columns.tolist()}")
print("DataFrame:")
print(df3)
print()

# Test 4: Let's see what happens with a tab separator
csv_tab = "number\tvalue\n1,000\t2,500\n3,750\t4,250\n"
print("=== Test 4: CSV with tab separator ===")
print("CSV content (with tabs):")
print(csv_tab.replace('\t', '[TAB]'))
df4 = read_csv(io.StringIO(csv_tab), sep="\t", thousands=",")
print(f"Shape: {df4.shape}")
print(f"Columns: {df4.columns.tolist()}")
print("DataFrame:")
print(df4)
print()

# Test 5: What if we explicitly set both sep=',' and thousands=','?
print("=== Test 5: Explicit sep=',' and thousands=',' ===")
csv_simple = "col1,col2\n1000,2000\n3000,4000\n"
print("CSV content (no thousands separators in data):")
print(csv_simple)
df5 = read_csv(io.StringIO(csv_simple), sep=",", thousands=",")
print(f"Shape: {df5.shape}")
print(f"Columns: {df5.columns.tolist()}")
print("DataFrame:")
print(df5)
print()

# Test 6: What happens to the interpretation?
print("=== Test 6: Understanding the parsing ===")
csv_debug = "number\n1,000\n"
print("When parsing '1,000' with sep=',' and thousands=',':")
print("1. First, the line '1,000' is split by sep=','")
print("2. This gives us two fields: '1' and '000'")
print("3. '1' becomes the index (parsed as row number)")
print("4. '000' becomes the value in the 'number' column")
print("5. With thousands=',', '000' is parsed as integer 0")
df6 = read_csv(io.StringIO(csv_debug), sep=",", thousands=",")
print(f"\nResult: {df6}")
print(f"Index values: {df6.index.tolist()}")
print(f"Column 'number' values: {df6['number'].tolist()}")