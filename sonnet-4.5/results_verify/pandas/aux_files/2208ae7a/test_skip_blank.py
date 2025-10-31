import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from io import StringIO

# Test with different skip_blank_lines settings
csv_with_empty_string = 'col0\n""\n'

print("Testing with skip_blank_lines=True (default):")
df_c_true = pd.read_csv(StringIO(csv_with_empty_string), engine='c', skip_blank_lines=True)
df_python_true = pd.read_csv(StringIO(csv_with_empty_string), engine='python', skip_blank_lines=True)
print(f"C engine shape: {df_c_true.shape}")
print(f"Python engine shape: {df_python_true.shape}")

print("\nTesting with skip_blank_lines=False:")
df_c_false = pd.read_csv(StringIO(csv_with_empty_string), engine='c', skip_blank_lines=False)
df_python_false = pd.read_csv(StringIO(csv_with_empty_string), engine='python', skip_blank_lines=False)
print(f"C engine shape: {df_c_false.shape}")
print(f"Python engine shape: {df_python_false.shape}")

# Test with actual blank line vs quoted empty string
print("\n" + "="*50)
print("Testing actual blank line:")
csv_blank_line = 'col0\n\n'
print(f"CSV with blank line: {repr(csv_blank_line)}")

df_c_blank = pd.read_csv(StringIO(csv_blank_line), engine='c', skip_blank_lines=True)
df_python_blank = pd.read_csv(StringIO(csv_blank_line), engine='python', skip_blank_lines=True)
print(f"C engine shape: {df_c_blank.shape}")
print(f"Python engine shape: {df_python_blank.shape}")

# Test with multiple quoted empty strings
print("\n" + "="*50)
print("Testing multiple quoted empty strings:")
csv_multiple = 'col0,col1\n"",""\n'
print(f"CSV: {repr(csv_multiple)}")

df_c_multi = pd.read_csv(StringIO(csv_multiple), engine='c', skip_blank_lines=True)
df_python_multi = pd.read_csv(StringIO(csv_multiple), engine='python', skip_blank_lines=True)
print(f"C engine shape: {df_c_multi.shape}")
print(f"Python engine shape: {df_python_multi.shape}")
print(f"\nC engine DataFrame:")
print(df_c_multi)
print(f"\nPython engine DataFrame:")
print(df_python_multi)