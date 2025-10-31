import pandas as pd
import io

# Test the exact example from the bug report
csv_str = 'col0\n""\na\n""\n'
print(f"Testing CSV string: {repr(csv_str)}")

df_c = pd.read_csv(io.StringIO(csv_str), engine='c')
df_python = pd.read_csv(io.StringIO(csv_str), engine='python')

print("C engine result:")
print(df_c)
print(f"Shape: {df_c.shape}")

print("\nPython engine result:")
print(df_python)
print(f"Shape: {df_python.shape}")

# Test with skip_blank_lines=False as mentioned in the bug report
print("\n" + "="*50)
print("Testing with skip_blank_lines=False:")

df_c_no_skip = pd.read_csv(io.StringIO(csv_str), engine='c', skip_blank_lines=False)
df_python_no_skip = pd.read_csv(io.StringIO(csv_str), engine='python', skip_blank_lines=False)

print("\nC engine result (skip_blank_lines=False):")
print(df_c_no_skip)
print(f"Shape: {df_c_no_skip.shape}")

print("\nPython engine result (skip_blank_lines=False):")
print(df_python_no_skip)
print(f"Shape: {df_python_no_skip.shape}")

try:
    pd.testing.assert_frame_equal(df_c_no_skip, df_python_no_skip, check_dtype=True)
    print("\nWith skip_blank_lines=False: DataFrames are equal")
except AssertionError as e:
    print(f"\nWith skip_blank_lines=False: DataFrames are NOT equal: {e}")

# Also test what actually constitutes a blank line
print("\n" + "="*50)
print("Testing truly blank lines vs quoted empty strings:")

# Test with truly blank lines
csv_blank = 'col0\n\na\n\n'
print(f"\nCSV with truly blank lines: {repr(csv_blank)}")

df_c_blank = pd.read_csv(io.StringIO(csv_blank), engine='c')
df_python_blank = pd.read_csv(io.StringIO(csv_blank), engine='python')

print("C engine result (blank lines):")
print(df_c_blank)
print(f"Shape: {df_c_blank.shape}")

print("\nPython engine result (blank lines):")
print(df_python_blank)
print(f"Shape: {df_python_blank.shape}")

# Test mixed case
csv_mixed = 'col0\n""\n\na\n'
print(f"\nCSV with mixed (quoted empty and blank): {repr(csv_mixed)}")

df_c_mixed = pd.read_csv(io.StringIO(csv_mixed), engine='c')
df_python_mixed = pd.read_csv(io.StringIO(csv_mixed), engine='python')

print("C engine result (mixed):")
print(df_c_mixed)
print(f"Shape: {df_c_mixed.shape}")

print("\nPython engine result (mixed):")
print(df_python_mixed)
print(f"Shape: {df_python_mixed.shape}")