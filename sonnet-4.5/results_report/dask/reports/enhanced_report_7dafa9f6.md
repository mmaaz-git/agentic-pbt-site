# Bug Report: dask.dataframe.io.parquet _aggregate_stats Missing 'name' Field in Column Statistics

**Target**: `dask.dataframe.io.parquet.utils._aggregate_stats`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_aggregate_stats` function creates column statistics dictionaries without the required 'name' field when encountering columns with uniform values (min == max) and null values, causing KeyError crashes in downstream code that expects this field to always be present.

## Property-Based Test

```python
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
```

<details>

<summary>
**Failing input**: `col_name='0', value=0, null_count=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 30, in <module>
    test_aggregate_stats_has_name_field()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 8, in test_aggregate_stats_has_name_field
    col_name=st.text(min_size=1, max_size=20),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 27, in test_aggregate_stats_has_name_field
    assert "name" in col_stat, f"Column stat missing 'name' field: {col_stat}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Column stat missing 'name' field: {'null_count': 1}
Falsifying example: test_aggregate_stats_has_name_field(
    # The test always failed when commented parts were varied together.
    col_name='0',  # or any other generated value
    value=0,  # or any other generated value
    null_count=1,  # or any other generated value
)

You can reproduce this example by temporarily adding @reproduce_failure('6.139.2', b'AIEwQQBBAQ==') as a decorator on your test case
```
</details>

## Reproducing the Bug

```python
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
```

<details>

<summary>
KeyError: 'name' when calling sorted_columns
</summary>
```
Column statistics returned by _aggregate_stats:
{'null_count': 10}

KeyError when calling sorted_columns: 'name'
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/repo.py", line 26, in <module>
    sorted_cols = sorted_columns(statistics, columns=columns_to_sort)
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 419, in sorted_columns
    if columns and c["name"] not in columns:
                   ~^^^^^^^^
KeyError: 'name'
```
</details>

## Why This Is A Bug

The `_aggregate_stats` function violates an implicit contract about the structure of column statistics dictionaries. When processing row-group statistics, the function has two code paths that create inconsistent output structures:

1. **Normal path** (lines 505-511, 520-526 in utils.py): Creates complete column statistics with all fields: `{"name": name, "min": minval, "max": maxval, "null_count": null_count}`

2. **"Dangerous stats" path** (lines 503, 518 in utils.py): When `minval == maxval and null_count > 0`, creates incomplete statistics: `{"null_count": null_count}` - missing the 'name' field.

This inconsistency causes crashes in downstream consumers like `sorted_columns` (core.py line 419) which unconditionally accesses `c["name"]` for all column statistics when the `columns` parameter is provided. The code comment "Remove dangerous stats" suggests the intent was to exclude unreliable min/max values, but the implementation incorrectly also removes the essential 'name' field that identifies which column the statistics belong to.

## Relevant Context

This bug affects Parquet file reading in Dask when specific data patterns are encountered:
- Boolean columns with uniform values (e.g., all True/False) plus nulls
- Categorical columns with a single category value plus nulls
- Timestamp columns where all non-null values are identical
- Any numeric column where min equals max and nulls are present

The bug is in an internal utility function (`_aggregate_stats`) that aggregates row-group statistics during Parquet file reading. While not part of the public API, this function is critical to Dask's Parquet reading pipeline.

Relevant code locations:
- Bug location: `/dask/dataframe/io/parquet/utils.py` lines 503 and 518
- Crash location: `/dask/dataframe/io/parquet/core.py` line 419 in `sorted_columns` function
- Documentation: The function lacks explicit documentation about the expected output format

## Proposed Fix

```diff
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