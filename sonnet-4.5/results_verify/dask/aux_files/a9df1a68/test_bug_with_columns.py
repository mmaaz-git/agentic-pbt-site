#!/usr/bin/env python3
"""Test script to reproduce the bug with columns parameter"""

from dask.dataframe.io.parquet.utils import _aggregate_stats
from dask.dataframe.io.parquet.core import sorted_columns

print("Testing sorted_columns with columns parameter...")
print("-" * 50)

# Test case from the bug report
file_path = "test.parquet"
file_row_group_stats = [{"num-rows": 100, "total_byte_size": 1000}]
file_row_group_column_stats = [[5, 5, 10]]  # min=5, max=5, null_count=10
stat_col_indices = ["x"]

result = _aggregate_stats(
    file_path,
    file_row_group_stats,
    file_row_group_column_stats,
    stat_col_indices
)

print(f"Result from _aggregate_stats:")
print(f"  Columns: {result['columns']}")

statistics = [result]

print("\nTest 1: sorted_columns without columns parameter:")
try:
    sorted_cols = sorted_columns(statistics)
    print(f"  ✓ Succeeded (returns empty list): {sorted_cols}")
except KeyError as e:
    print(f"  ✗ KeyError raised: {e}")

print("\nTest 2: sorted_columns WITH columns parameter=['x']:")
try:
    sorted_cols = sorted_columns(statistics, columns=["x"])
    print(f"  ✓ Succeeded: {sorted_cols}")
except KeyError as e:
    print(f"  ✗ KeyError raised: {e}")
    print("     This is the bug! The code tries to access c['name'] when columns is not None")