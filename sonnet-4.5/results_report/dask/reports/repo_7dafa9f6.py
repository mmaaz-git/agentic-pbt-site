import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.dataframe.io.parquet.utils import _aggregate_stats
from dask.dataframe.io.parquet.core import sorted_columns

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

print("Column statistics returned by _aggregate_stats:")
print(result["columns"][0])
print()

try:
    statistics = [result]
    columns_to_sort = ["x"]
    sorted_cols = sorted_columns(statistics, columns=columns_to_sort)
    print("sorted_columns succeeded")
except KeyError as e:
    print(f"KeyError when calling sorted_columns: {e}")
    import traceback
    traceback.print_exc()