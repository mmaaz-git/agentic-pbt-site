#!/usr/bin/env python3
"""Test script to reproduce the downstream impact of the bug"""

from dask.dataframe.io.parquet.utils import _aggregate_stats
from dask.dataframe.io.parquet.core import sorted_columns

print("Reproducing the bug and its downstream impact...")
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
if result['columns']:
    print(f"  First column stat: {result['columns'][0]}")

print("\nNow attempting to use sorted_columns with this result...")
statistics = [result]

try:
    sorted_cols = sorted_columns(statistics)
    print(f"sorted_columns succeeded: {sorted_cols}")
except KeyError as e:
    print(f"✗ KeyError raised: {e}")
    print("  This demonstrates the downstream impact of the missing 'name' field")

print("\n" + "-" * 50)
print("Testing normal case (where min != max):")
# Test a normal case where min != max
file_row_group_column_stats_normal = [[1, 10, 5]]  # min=1, max=10, null_count=5
result_normal = _aggregate_stats(
    file_path,
    file_row_group_stats,
    file_row_group_column_stats_normal,
    stat_col_indices
)

print(f"Result columns (normal case): {result_normal['columns']}")
if result_normal['columns']:
    print(f"First column stat: {result_normal['columns'][0]}")
    if 'name' in result_normal['columns'][0]:
        print("✓ 'name' field is present in normal case")

statistics_normal = [result_normal]
try:
    sorted_cols_normal = sorted_columns(statistics_normal)
    print(f"✓ sorted_columns succeeded for normal case: {sorted_cols_normal}")
except KeyError as e:
    print(f"✗ KeyError raised even in normal case: {e}")