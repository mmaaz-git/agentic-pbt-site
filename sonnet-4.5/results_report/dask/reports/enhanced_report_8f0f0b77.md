# Bug Report: dask.dataframe._aggregate_statistics_to_file crashes on None statistics

**Target**: `dask.dataframe.dask_expr.io.parquet._aggregate_statistics_to_file`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_aggregate_statistics_to_file` function crashes with an `AttributeError` when processing parquet file metadata that contains `None` values in the column `statistics` field, which is a valid state according to the Apache Parquet specification.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from dask.dataframe.dask_expr.io.parquet import _aggregate_statistics_to_file

stat_dict = st.fixed_dictionaries({
    "num_rows": st.integers(min_value=0, max_value=10000),
    "num_row_groups": st.integers(min_value=1, max_value=10),
    "serialized_size": st.integers(min_value=0, max_value=1000000),
    "row_groups": st.lists(
        st.fixed_dictionaries({
            "num_rows": st.integers(min_value=0, max_value=1000),
            "total_byte_size": st.integers(min_value=0, max_value=100000),
            "columns": st.lists(
                st.fixed_dictionaries({
                    "total_compressed_size": st.integers(min_value=0, max_value=10000),
                    "total_uncompressed_size": st.integers(min_value=0, max_value=10000),
                    "path_in_schema": st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('L',))),
                    "statistics": st.one_of(
                        st.none(),
                        st.fixed_dictionaries({
                            "min": st.integers(),
                            "max": st.integers(),
                            "null_count": st.integers(min_value=0, max_value=1000),
                            "num_values": st.integers(min_value=0, max_value=1000),
                            "distinct_count": st.integers(min_value=0, max_value=1000),
                        })
                    )
                }),
                min_size=1,
                max_size=5
            )
        }),
        min_size=1,
        max_size=5
    )
})

@given(st.lists(stat_dict, min_size=1, max_size=5))
@settings(max_examples=100)
def test_aggregate_statistics_preserves_num_files(stats):
    result = _aggregate_statistics_to_file(stats)
    assert len(result) == len(stats)

if __name__ == "__main__":
    test_aggregate_statistics_preserves_num_files()
```

<details>

<summary>
**Failing input**: `stats=[{'num_rows': 0, 'num_row_groups': 1, 'serialized_size': 0, 'row_groups': [{'num_rows': 0, 'total_byte_size': 0, 'columns': [{'total_compressed_size': 0, 'total_uncompressed_size': 0, 'path_in_schema': 'A', 'statistics': None}]}]}]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 47, in <module>
    test_aggregate_statistics_preserves_num_files()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 41, in test_aggregate_statistics_preserves_num_files
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 43, in test_aggregate_statistics_preserves_num_files
    result = _aggregate_statistics_to_file(stats)
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/parquet.py", line 1979, in _aggregate_statistics_to_file
    file_stat.update(_agg_dicts(file_stat.pop("row_groups"), agg_func))
                     ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/parquet.py", line 1931, in _agg_dicts
    result2[k] = agg(v)
                 ~~~^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/parquet.py", line 1948, in _aggregate_columns
    return [_agg_dicts(c, agg_cols) for c in combine]
            ~~~~~~~~~~^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/parquet.py", line 1931, in _agg_dicts
    result2[k] = agg(v)
                 ~~~^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/parquet.py", line 1922, in _agg_dicts
    for k, v in d.items():
                ^^^^^^^
AttributeError: 'NoneType' object has no attribute 'items'
Falsifying example: test_aggregate_statistics_preserves_num_files(
    stats=[{'num_rows': 0,
      'num_row_groups': 1,
      'serialized_size': 0,
      'row_groups': [{'num_rows': 0,
        'total_byte_size': 0,
        'columns': [{'total_compressed_size': 0,
          'total_uncompressed_size': 0,
          'path_in_schema': 'A',
          'statistics': None}]}]}],
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.dataframe.dask_expr.io.parquet import _aggregate_statistics_to_file

stats = [{
    'num_rows': 0,
    'num_row_groups': 1,
    'serialized_size': 0,
    'row_groups': [{
        'num_rows': 0,
        'total_byte_size': 0,
        'columns': [{
            'total_compressed_size': 0,
            'total_uncompressed_size': 0,
            'path_in_schema': 'A',
            'statistics': None
        }]
    }]
}]

result = _aggregate_statistics_to_file(stats)
```

<details>

<summary>
AttributeError: 'NoneType' object has no attribute 'items' at line 1922
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/repo.py", line 22, in <module>
    result = _aggregate_statistics_to_file(stats)
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/parquet.py", line 1979, in _aggregate_statistics_to_file
    file_stat.update(_agg_dicts(file_stat.pop("row_groups"), agg_func))
                     ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/parquet.py", line 1931, in _agg_dicts
    result2[k] = agg(v)
                 ~~~^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/parquet.py", line 1948, in _aggregate_columns
    return [_agg_dicts(c, agg_cols) for c in combine]
            ~~~~~~~~~~^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/parquet.py", line 1931, in _agg_dicts
    result2[k] = agg(v)
                 ~~~^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/parquet.py", line 1922, in _agg_dicts
    for k, v in d.items():
                ^^^^^^^
AttributeError: 'NoneType' object has no attribute 'items'
```
</details>

## Why This Is A Bug

The Apache Parquet specification explicitly allows column statistics to be absent (represented as `None` in Python). According to the specification, statistics are optional metadata that can be omitted for performance reasons or when not applicable. The `_aggregate_statistics_to_file` function fails to handle this valid case, causing a crash when it tries to call `.items()` on a `None` value.

This contradicts the expected behavior in multiple ways:

1. **Parquet specification compliance**: The function should handle all valid parquet metadata structures, including those with missing statistics.

2. **Inconsistent with other code**: The same codebase contains the `_extract_stats` function (lines 1911-1914) which explicitly checks for and handles `None` statistics by skipping them with a `continue` statement. This shows that the developers are aware that statistics can be `None`.

3. **Common real-world scenario**: Parquet files without statistics are frequently encountered in production:
   - Writers may disable statistics for performance optimization
   - Large string or binary columns often omit statistics to avoid memory overhead
   - Legacy parquet writers may not generate statistics
   - Certain data types may not support statistics

4. **Unhelpful error message**: The AttributeError provides no indication that missing statistics are the cause, making debugging difficult for users who encounter this issue.

## Relevant Context

The crash occurs in the `_agg_dicts` function at line 1922 when it attempts to iterate over dictionary items:

```python
def _agg_dicts(dicts, agg_funcs):
    result = {}
    for d in dicts:
        for k, v in d.items():  # Line 1922 - crashes here when d is None
```

The function is called as part of aggregating column statistics through this call chain:
1. `_aggregate_statistics_to_file` aggregates row groups
2. `_aggregate_columns` aggregates columns across row groups
3. `_agg_dicts` is called with `statistics` as the aggregation target
4. When `statistics` is `None`, the function crashes trying to call `.items()` on `None`

Related code in the same file shows awareness of `None` statistics:
- `_extract_stats` function (line 1911): `if col["statistics"] is None: continue`
- This indicates the issue is an oversight rather than intentional design

Documentation references:
- Apache Parquet Format Specification: https://parquet.apache.org/docs/file-format/metadata/
- Dask source code: https://github.com/dask/dask/blob/main/dask/dataframe/dask_expr/io/parquet.py

## Proposed Fix

```diff
--- a/dask/dataframe/dask_expr/io/parquet.py
+++ b/dask/dataframe/dask_expr/io/parquet.py
@@ -1919,6 +1919,8 @@ def _extract_stats(original):
 def _agg_dicts(dicts, agg_funcs):
     result = {}
     for d in dicts:
+        if d is None:
+            continue
         for k, v in d.items():
             if k not in result:
                 result[k] = [v]
```