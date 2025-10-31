# Bug Report: _aggregate_statistics_to_file crashes with None statistics

**Target**: `dask.dataframe.dask_expr.io.parquet._aggregate_statistics_to_file`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_aggregate_statistics_to_file` function crashes with an `AttributeError` when processing parquet file metadata that contains `None` values in the `statistics` field, which is a valid state according to the parquet specification.

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
```

**Failing input**:
```python
stats=[{
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
```

## Reproducing the Bug

```python
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

**Error**:
```
AttributeError: 'NoneType' object has no attribute 'items'
  File "dask/dataframe/dask_expr/io/parquet.py", line 1922, in _agg_dicts
    for k, v in d.items():
                ^^^^^^^
```

## Why This Is A Bug

1. **None statistics are valid**: The parquet specification allows columns to have no statistics, and the code itself acknowledges this in the `_extract_stats` function (line 1911-1914) where it explicitly checks for and handles `None` statistics.

2. **Inconsistent handling**: The `_extract_stats` function skips `None` statistics with a `continue` statement, but `_agg_dicts` doesn't handle them, causing a crash downstream.

3. **Real-world impact**: Parquet files with missing statistics are common in practice, especially for:
   - Files written with statistics disabled for performance
   - Large string columns where statistics are expensive to compute
   - Legacy parquet files

## Fix

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