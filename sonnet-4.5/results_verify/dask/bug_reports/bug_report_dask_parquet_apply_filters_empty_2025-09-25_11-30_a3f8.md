# Bug Report: dask.dataframe.io.parquet apply_filters Empty List

**Target**: `dask.dataframe.io.parquet.core.apply_filters`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `apply_filters` function crashes with an `IndexError` when given an empty filters list, instead of returning all parts as expected.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import dask.dataframe.io.parquet.core as parquet_core

@given(
    st.lists(st.text(min_size=1), min_size=1, max_size=10),
    st.lists(st.dictionaries(
        st.text(min_size=1),
        st.one_of(st.none(), st.integers(), st.floats(allow_nan=False)),
        min_size=0,
        max_size=10
    ), min_size=1, max_size=10)
)
def test_apply_filters_returns_subset(parts, statistics):
    assume(len(parts) == len(statistics))
    for stats in statistics:
        stats['columns'] = []
    filters = []
    out_parts, out_statistics = parquet_core.apply_filters(parts, statistics, filters)
    assert len(out_parts) <= len(parts)
```

**Failing input**: `parts=['part1'], statistics=[{}], filters=[]`

## Reproducing the Bug

```python
import dask.dataframe.io.parquet.core as parquet_core

parts = ['part1']
statistics = [{'columns': []}]
filters = []

out_parts, out_statistics = parquet_core.apply_filters(parts, statistics, filters)
```

## Why This Is A Bug

The function attempts to access `filters[0]` without checking if the list is empty. According to the docstring, the function should return "the same as the input, but possibly a subset", so an empty filter list should logically return all inputs unfiltered. The crash occurs at line 556 when unpacking: `conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]`

## Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -553,6 +553,9 @@ def apply_filters(parts, statistics, filters):

         return parts, statistics

+    if not filters:
+        return parts, statistics
+
     conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]

     out_parts, out_statistics = apply_conjunction(parts, statistics, conjunction)