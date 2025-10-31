#!/usr/bin/env python3
"""Test how pandas handles index_col with columns parameter in read_csv"""

import pandas as pd
import tempfile
import os

# Create a test CSV file
tmp_file = tempfile.mktemp(suffix='.csv')

# Create test data
df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9]
})

# Save to CSV
df.to_csv(tmp_file, index=False)

print("Original data saved to CSV:")
print(df)

try:
    # Try reading with columns=['a', 'b'] and index_col='c'
    # This tests if pandas automatically includes index_col when not in columns
    print("\n=== Test 1: columns=['a', 'b'], index_col='c' ===")
    df_read = pd.read_csv(tmp_file, usecols=['a', 'b'], index_col='c')
    print("Success! Result:")
    print(df_read)
except Exception as e:
    print(f"Error: {e}")

try:
    # Test with explicit columns including index
    print("\n=== Test 2: columns=['a', 'b', 'c'], index_col='c' ===")
    df_read = pd.read_csv(tmp_file, usecols=['a', 'b', 'c'], index_col='c')
    print("Success! Result:")
    print(df_read)
except Exception as e:
    print(f"Error: {e}")

# Clean up
os.unlink(tmp_file)