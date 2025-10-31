# Bug Report: dask.dataframe.io.parquet sorted_columns IndexError

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with an `IndexError` when given statistics with mismatched column counts across row groups.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import dask.dataframe.io.parquet.core as parquet_core

@given(st.lists(
    st.fixed_dictionaries({
        'columns': st.lists(
            st.fixed_dictionaries({
                'name': st.text(min_size=1, max_size=20),
                'min': st.one_of(st.none(), st.integers(-1000, 1000)),
                'max': st.one_of(st.none(), st.integers(-1000, 1000))
            }),
            min_size=1,
            max_size=5
        )
    }),
    min_size=1,
    max_size=10
))
def test_sorted_columns_divisions_are_sorted(statistics):
    result = parquet_core.sorted_columns(statistics)
    for item in result:
        assert item['divisions'] == sorted(item['divisions'])
```

**Failing input**:
```python
statistics = [
    {'columns': [
        {'name': 'col1', 'min': 0, 'max': 10},
        {'name': 'col2', 'min': 0, 'max': 10}
    ]},
    {'columns': [
        {'name': 'col1', 'min': 11, 'max': 20}
    ]}
]
```

## Reproducing the Bug

```python
import dask.dataframe.io.parquet.core as parquet_core

statistics = [
    {'columns': [
        {'name': 'col1', 'min': 0, 'max': 10},
        {'name': 'col2', 'min': 0, 'max': 10}
    ]},
    {'columns': [
        {'name': 'col1', 'min': 11, 'max': 20}
    ]}
]

result = parquet_core.sorted_columns(statistics)
```

## Why This Is A Bug

The function iterates over columns in the first statistics entry using an index, but then accesses that same index in subsequent statistics entries without checking if they have the same number of columns. This causes an `IndexError: list index out of range` when statistics have different column counts, which can legitimately occur when processing heterogeneous parquet datasets.

## Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -417,6 +417,11 @@ def sorted_columns(statistics, columns=None):
     for i, c in enumerate(statistics[0]["columns"]):
         if columns and c["name"] not in columns:
             continue
+        # Check if all statistics have this column index
+        if not all(i < len(s["columns"]) for s in statistics):
+            continue
+        # Check if column names match across all statistics
+        if not all(s["columns"][i]["name"] == c["name"] for s in statistics):
+            continue
         if not all(
             "min" in s["columns"][i] and "max" in s["columns"][i] for s in statistics
         ):