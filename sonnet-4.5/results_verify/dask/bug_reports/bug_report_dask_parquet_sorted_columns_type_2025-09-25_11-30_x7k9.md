# Bug Report: dask.dataframe.io.parquet sorted_columns TypeError

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with a `TypeError` when comparing integer min values with None max values in column statistics.

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
    {'columns': [{'name': 'col1', 'min': None, 'max': None}]},
    {'columns': [{'name': 'col1', 'min': 0, 'max': None}]}
]
```

## Reproducing the Bug

```python
import dask.dataframe.io.parquet.core as parquet_core

statistics = [
    {'columns': [{'name': 'col1', 'min': None, 'max': None}]},
    {'columns': [{'name': 'col1', 'min': 0, 'max': None}]}
]

result = parquet_core.sorted_columns(statistics)
```

## Why This Is A Bug

The function has a check `if c["min"] is None:` to handle None min values, but it doesn't check if `max` is None before using it in the comparison `if c["min"] >= max:`. This causes a TypeError when min is a number but max is None. This can occur with incomplete or corrupted parquet statistics.

## Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -428,6 +428,8 @@ def sorted_columns(statistics, columns=None):
             if c["min"] is None:
                 success = False
                 break
+            if max is None or c["max"] is None:
+                success = False
+                break
             if c["min"] >= max:
                 divisions.append(c["min"])
                 max = c["max"]