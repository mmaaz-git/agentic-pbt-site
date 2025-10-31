# Bug Report: dask.dataframe.io.parquet _aggregate_stats Missing Column Name

**Target**: `dask.dataframe.io.parquet.utils._aggregate_stats`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_aggregate_stats` function in `dask/dataframe/io/parquet/utils.py` creates column statistics dictionaries without a 'name' field when `min == max` and `null_count > 0`. This causes KeyError when downstream code attempts to access the column name.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.parquet.utils import _aggregate_stats

@given(
    col_name=st.text(min_size=1, max_size=20),
    value=st.integers(min_value=-100, max_value=100),
    null_count=st.integers(min_value=1, max_value=100),
)
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
```

**Failing input**: `col_name="x"`, `value=5`, `null_count=10`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/dask')

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

print(result["columns"][0])

statistics = [result]
sorted_columns(statistics)
```

Output:
```
{'null_count': 10}
Traceback (most recent call last):
  ...
KeyError: 'name'
```

## Why This Is A Bug

The function `_aggregate_stats` is responsible for aggregating row-group statistics. When it encounters a "dangerous" case where `min == max` and there are null values, it creates a column stats dictionary with only `{'null_count': null_count}` (lines 503 and 518 in utils.py).

However, downstream code such as `sorted_columns` (core.py line 419) expects all column statistics to have a 'name' field:

```python
for i, c in enumerate(statistics[0]["columns"]):
    if columns and c["name"] not in columns:
        ...
```

This causes a KeyError when code tries to access `col["name"]` on the incomplete column statistics.

## Fix

```diff
diff --git a/dask/dataframe/io/parquet/utils.py b/dask/dataframe/io/parquet/utils.py
index 1234567..abcdefg 100644
--- a/dask/dataframe/io/parquet/utils.py
+++ b/dask/dataframe/io/parquet/utils.py
@@ -500,7 +500,7 @@ def _aggregate_stats(
                 null_count = file_row_group_column_stats[0][i + 2]
                 if minval == maxval and null_count:
                     # Remove "dangerous" stats (min == max, but null values exist)
-                    s["columns"].append({"null_count": null_count})
+                    s["columns"].append({"name": name, "null_count": null_count})
                 else:
                     s["columns"].append(
                         {
@@ -515,7 +515,7 @@ def _aggregate_stats(
                 maxval = df_cols.iloc[:, i + 1].dropna().max()
                 null_count = df_cols.iloc[:, i + 2].sum()
                 if minval == maxval and null_count:
-                    s["columns"].append({"null_count": null_count})
+                    s["columns"].append({"name": name, "null_count": null_count})
                 else:
                     s["columns"].append(
                         {
```