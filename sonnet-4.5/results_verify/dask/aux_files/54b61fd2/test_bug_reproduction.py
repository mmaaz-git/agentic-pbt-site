#!/usr/bin/env python3
"""Test script to reproduce the sorted_columns bug"""

import dask.dataframe.io.parquet.core as parquet_core

# Test case from bug report
statistics = [
    {'columns': [{'name': 'col1', 'min': None, 'max': None}]},
    {'columns': [{'name': 'col1', 'min': 0, 'max': None}]}
]

print("Testing sorted_columns with statistics:")
for i, stat in enumerate(statistics):
    print(f"  Row {i}: {stat}")

try:
    result = parquet_core.sorted_columns(statistics)
    print(f"\nResult: {result}")
except TypeError as e:
    print(f"\nTypeError occurred: {e}")
    print("Bug confirmed: TypeError when comparing integer with None")
except Exception as e:
    print(f"\nUnexpected error: {type(e).__name__}: {e}")