from pandas.core.computation.parsing import clean_column_name

# Test with null byte
name = '\x00'
print(f"Testing clean_column_name with null byte: repr(name) = {repr(name)}")

try:
    result = clean_column_name(name)
    print(f"Result: {repr(result)}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Also verify that pandas DataFrames can have null byte column names
import pandas as pd
print("\nVerifying pandas DataFrame can have null byte column names:")
df = pd.DataFrame({'\x00': [1, 2, 3]})
print(f"DataFrame created with null byte column name: {list(df.columns)}")
print(f"Column name repr: {repr(df.columns[0])}")