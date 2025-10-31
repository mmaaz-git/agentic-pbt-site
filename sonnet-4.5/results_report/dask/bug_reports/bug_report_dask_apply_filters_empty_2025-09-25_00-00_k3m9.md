# Bug Report: dask.dataframe.io.parquet apply_filters IndexError on Empty Filters

**Target**: `dask.dataframe.io.parquet.core.apply_filters`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `apply_filters` function crashes with an `IndexError` when called with an empty `filters` list, even though this is a valid input according to the function's documented behavior.

## Property-Based Test

```python
from hypothesis import given, assume, strategies as st
import dask.dataframe.io.parquet.core as core

@given(
    st.lists(st.text(), min_size=0, max_size=20),
    st.lists(st.dictionaries(st.text(), st.integers()), min_size=0, max_size=20)
)
def test_apply_filters_empty_filters_returns_all(parts, statistics):
    assume(len(parts) == len(statistics))

    filters = []

    filtered_parts, filtered_stats = core.apply_filters(parts, statistics, filters)

    assert filtered_parts == parts
    assert filtered_stats == statistics
```

**Failing input**: `parts=[], statistics=[], filters=[]`

## Reproducing the Bug

```python
import dask.dataframe.io.parquet.core as core

parts = []
statistics = []
filters = []

filtered_parts, filtered_stats = core.apply_filters(parts, statistics, filters)
```

This raises:
```
IndexError: list index out of range
```

## Why This Is A Bug

The function's docstring states that it should return "the same as the input, but possibly a subset". When no filters are provided (empty list), the expected behavior is to return all parts and statistics unchanged. This is a reasonable use case - users may conditionally apply filters, and when there are no filters to apply, the function should handle this gracefully.

The crash occurs because the code tries to access `filters[0]` without first checking if the list is empty:

```python
conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
```

## Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -553,7 +553,10 @@ def apply_filters(parts, statistics, filters):

         return parts, statistics

-    conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
+    if not filters:
+        return parts, statistics
+
+    conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]

     out_parts, out_statistics = apply_conjunction(parts, statistics, conjunction)
     for conjunction in disjunction:
```