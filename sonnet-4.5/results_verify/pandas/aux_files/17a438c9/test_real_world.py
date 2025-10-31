#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas as pd
from io import StringIO

print("Test: Creating DataFrame with duplicate NaN column names")
print("=" * 60)

csv = StringIO("1,2\n3,4")
try:
    df = pd.read_csv(csv, names=[float('nan'), float('nan')])
    print("DataFrame created successfully with duplicate NaN column names!")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    print("\nDataFrame content:")
    print(df)
    print("\nColumn access test:")
    print(f"Accessing column NaN: {df[float('nan')]}")
except ValueError as e:
    print(f"Failed to create DataFrame: {e}")

print("\n" + "=" * 60)
print("\nTest: Creating DataFrame with regular duplicate names (control)")
csv2 = StringIO("1,2\n3,4")
try:
    df2 = pd.read_csv(csv2, names=['col1', 'col1'])
    print("DataFrame created with duplicate regular names!")
    print(f"DataFrame columns: {df2.columns.tolist()}")
except ValueError as e:
    print(f"Failed to create DataFrame as expected: {e}")