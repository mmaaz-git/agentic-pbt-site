#!/usr/bin/env python3
"""Test to verify the actual data corruption in Dask DataFrames"""

import numpy as np
import pandas as pd
from dask.dataframe.dask_expr.io.io import FromArray
import dask.dataframe as dd

print("Testing data corruption in Dask DataFrame creation")
print("=" * 60)

# Create a simple array with distinct values
arr = np.array([[10, 20, 30], [40, 50, 60]])
original_columns = ['a', 'b', 'c']
requested_columns = ['c', 'a']

print("Original numpy array:")
print(arr)
print(f"Original columns: {original_columns}")
print(f"Requested columns: {requested_columns}")
print()

# Test with pandas first (expected behavior)
print("Pandas DataFrame (expected behavior):")
pdf = pd.DataFrame(arr, columns=original_columns)
print("Original DataFrame:")
print(pdf)
print("\nSelecting columns ['c', 'a']:")
pdf_selected = pdf[requested_columns]
print(pdf_selected)
print()

# Now test with FromArray directly
from_array = FromArray(
    frame=arr,
    chunksize=10,
    original_columns=original_columns,
    meta=None,
    columns=requested_columns
)

print("FromArray investigation:")
print(f"  _column_indices: {from_array._column_indices}")
print(f"  Meta columns: {from_array._meta.columns.tolist() if hasattr(from_array._meta, 'columns') else 'N/A'}")
print()

# Test actual Dask DataFrame creation
print("Testing with dask.dataframe.from_array:")
ddf = dd.from_array(arr, columns=original_columns, chunksize=10)
print(f"Original Dask DataFrame columns: {ddf.columns.tolist()}")
print("Original Dask DataFrame (computed):")
print(ddf.compute())
print()

# Select columns in specific order
ddf_selected = ddf[requested_columns]
print(f"Selected columns ['c', 'a'] - Dask DataFrame columns: {ddf_selected.columns.tolist()}")
print("Selected Dask DataFrame (computed):")
result = ddf_selected.compute()
print(result)
print()

# Check if the data matches
print("Data Corruption Check:")
print("-" * 40)
print(f"Expected column 'c' values: {pdf_selected['c'].tolist()}")
print(f"Actual column 'c' values:   {result['c'].tolist()}")
print(f"Match: {pdf_selected['c'].tolist() == result['c'].tolist()}")
print()
print(f"Expected column 'a' values: {pdf_selected['a'].tolist()}")
print(f"Actual column 'a' values:   {result['a'].tolist()}")
print(f"Match: {pdf_selected['a'].tolist() == result['a'].tolist()}")
print()

if pdf_selected.equals(result):
    print("✓ No data corruption - values match expected")
else:
    print("✗ DATA CORRUPTION DETECTED - values don't match!")
    print("\nDifference:")
    print("Expected:")
    print(pdf_selected)
    print("\nGot:")
    print(result)