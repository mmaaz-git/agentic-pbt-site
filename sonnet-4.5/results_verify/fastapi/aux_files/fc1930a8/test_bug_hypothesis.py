#!/usr/bin/env python3
"""Test script to reproduce the dask parquet bug using hypothesis"""

from hypothesis import given, strategies as st, settings
from dask.dataframe.io.parquet.utils import _aggregate_stats

@given(
    col_name=st.text(min_size=1, max_size=20),
    value=st.integers(min_value=-100, max_value=100),
    null_count=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=10)
def test_aggregate_stats_has_name_field(col_name, value, null_count):
    file_path = "test.parquet"
    file_row_group_stats = [{"num-rows": 100, "total_byte_size": 1000}]
    file_row_group_column_stats = [[value, value, null_count]]
    stat_col_indices = [col_name]

    result = _aggregate_stats(
        file_path,
        file_row_group_stats,
        file_row_group_column_stats,
        stat_col_indices
    )

    for col_stat in result["columns"]:
        assert "name" in col_stat, f"Column stat missing 'name' field: {col_stat}"

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_aggregate_stats_has_name_field()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Test specific failing input mentioned in the bug report
    print("\nTesting specific failing input (col_name='x', value=5, null_count=10):")
    file_path = "test.parquet"
    file_row_group_stats = [{"num-rows": 100, "total_byte_size": 1000}]
    file_row_group_column_stats = [[5, 5, 10]]
    stat_col_indices = ["x"]

    result = _aggregate_stats(
        file_path,
        file_row_group_stats,
        file_row_group_column_stats,
        stat_col_indices
    )

    print(f"Result columns: {result['columns']}")
    if result['columns']:
        print(f"First column stat: {result['columns'][0]}")
        if 'name' in result['columns'][0]:
            print("✓ 'name' field is present")
        else:
            print("✗ 'name' field is MISSING!")