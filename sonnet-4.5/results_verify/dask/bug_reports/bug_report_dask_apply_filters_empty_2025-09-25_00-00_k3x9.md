# Bug Report: dask.dataframe.io.parquet.core.apply_filters IndexError on Empty Filters

**Target**: `dask.dataframe.io.parquet.core.apply_filters`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `apply_filters` function crashes with an `IndexError` when passed an empty filters list, despite empty filters being a valid input that should return all parts unfiltered.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.parquet.core import apply_filters

@given(st.lists(st.dictionaries(
    st.text(min_size=1, max_size=10),
    st.one_of(st.integers(), st.floats(allow_nan=False), st.text())
)))
def test_apply_filters_empty_filters_identity(parts):
    statistics = [{"columns": []} for _ in parts]
    filters = []
    result_parts, result_stats = apply_filters(parts, statistics, filters)
    assert len(result_parts) == len(parts)
    assert len(result_stats) == len(statistics)
```

**Failing input**: `parts=[], statistics=[], filters=[]`

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import apply_filters

parts = []
statistics = []
filters = []

result_parts, result_stats = apply_filters(parts, statistics, filters)
```

**Output**:
```
IndexError: list index out of range
```

This also occurs with non-empty parts:
```python
parts = [{"piece": 1}, {"piece": 2}]
statistics = [{"columns": []}, {"columns": []}]
filters = []

result_parts, result_stats = apply_filters(parts, statistics, filters)
```

## Why This Is A Bug

Empty filters is a valid and reasonable input representing "no filtering" - the function should return all parts and statistics unchanged. The docstring does not require filters to be non-empty, and the function signature accepts an empty list. When users want to conditionally apply filters, they may naturally pass an empty list when no filters are needed.

The crash occurs at line 556:
```python
conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
```

This line attempts to access `filters[0]` without first checking if `filters` is empty.

## Fix

```diff
--- a/core.py
+++ b/core.py
@@ -553,6 +553,10 @@ def apply_filters(parts, statistics, filters):

         return parts, statistics

+    # Handle empty filters - return all parts unfiltered
+    if not filters:
+        return parts, statistics
+
     conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]

     out_parts, out_statistics = apply_conjunction(parts, statistics, conjunction)
```