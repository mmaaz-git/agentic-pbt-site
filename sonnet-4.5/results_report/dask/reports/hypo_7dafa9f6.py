import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from dask.dataframe.io.parquet.utils import _aggregate_stats

@given(
    col_name=st.text(min_size=1, max_size=20),
    value=st.integers(min_value=-100, max_value=100),
    null_count=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=10, print_blob=True)
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
    test_aggregate_stats_has_name_field()