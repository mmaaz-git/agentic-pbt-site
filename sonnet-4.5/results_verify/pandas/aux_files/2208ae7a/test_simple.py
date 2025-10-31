import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from io import StringIO

# Test the specific example from the bug report
csv_with_empty_string = 'col0\n""\n'

df_c = pd.read_csv(StringIO(csv_with_empty_string), engine='c')
df_python = pd.read_csv(StringIO(csv_with_empty_string), engine='python')

print(f"CSV content: {repr(csv_with_empty_string)}")
print(f"C engine shape: {df_c.shape}")
print(f"Python engine shape: {df_python.shape}")
print(f"\nC engine DataFrame:")
print(df_c)
print(f"\nPython engine DataFrame:")
print(df_python)

# Also test with the failing example from Hypothesis
print("\n" + "="*50)
print("Testing with DataFrame({'col0': ['']})")
test_df = pd.DataFrame({'col0': ['']})
csv_string = test_df.to_csv(index=False)
print(f"Generated CSV: {repr(csv_string)}")

df_c2 = pd.read_csv(StringIO(csv_string), engine='c')
df_python2 = pd.read_csv(StringIO(csv_string), engine='python')

print(f"C engine shape: {df_c2.shape}")
print(f"Python engine shape: {df_python2.shape}")
print(f"\nC engine DataFrame:")
print(df_c2)
print(f"\nPython engine DataFrame:")
print(df_python2)