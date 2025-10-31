#!/usr/bin/env python3
"""Test to understand how dd.from_map passes kwargs to functions"""

import dask.dataframe as dd
import pandas as pd

def test_function(x, *, columns=None):
    """Test function that mutates columns"""
    print(f"  Inside test_function: received columns={columns}, id={id(columns) if columns else None}")
    if columns is not None:
        columns.append("MUTATED")
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

print("=" * 60)
print("Testing dd.from_map behavior with columns kwarg")
print("=" * 60)

# Create a columns list
my_columns = ["col1", "col2"]
print(f"Original columns: {my_columns}, id={id(my_columns)}")

# Use dd.from_map with columns kwarg
df = dd.from_map(
    test_function,
    [1, 2, 3],  # 3 partitions
    columns=my_columns,
    meta=pd.DataFrame({"a": [0], "b": [0]})
)

print(f"After from_map creation: {my_columns}")

# Try to compute to see if mutation happens
print("\nComputing the DataFrame...")
try:
    result = df.compute()
    print(f"After compute: {my_columns}")
except Exception as e:
    print(f"Error during compute: {e}")
    print(f"After failed compute: {my_columns}")