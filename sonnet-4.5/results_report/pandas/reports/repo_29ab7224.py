import pandas as pd
import io

# Create a CSV string with quoted empty strings
csv_str = 'col0\n""\na\n""\n'

# Parse with C engine
df_c = pd.read_csv(io.StringIO(csv_str), engine='c')

# Parse with Python engine
df_python = pd.read_csv(io.StringIO(csv_str), engine='python')

print("CSV input string:")
print(repr(csv_str))
print()

print("C engine result:")
print(df_c)
print(f"Shape: {df_c.shape}")
print(f"Values: {df_c['col0'].tolist()}")
print()

print("Python engine result:")
print(df_python)
print(f"Shape: {df_python.shape}")
print(f"Values: {df_python['col0'].tolist()}")
print()

print("Are the DataFrames equal?")
try:
    pd.testing.assert_frame_equal(df_c, df_python, check_dtype=True)
    print("Yes, they are equal")
except AssertionError as e:
    print(f"No, they differ: {str(e)}")