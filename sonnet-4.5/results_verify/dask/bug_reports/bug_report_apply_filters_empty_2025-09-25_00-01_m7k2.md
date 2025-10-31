# Bug Report: apply_filters Empty List Crash

**Target**: `dask.dataframe.io.parquet.core.apply_filters`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`apply_filters` crashes with an `IndexError` when passed an empty list for the `filters` parameter. The function attempts to access `filters[0]` without first checking if the list is empty.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.dask_expr.io import apply_filters

@st.composite
def parts_and_stats(draw):
    num_parts = draw(st.integers(min_value=0, max_value=10))
    parts = [{"piece": (i,)} for i in range(num_parts)]
    stats = []
    for i in range(num_parts):
        stat = {
            f"col_{j}": {
                "min": draw(st.integers(min_value=-100, max_value=100)),
                "max": draw(st.integers(min_value=-100, max_value=100))
            }
            for j in range(draw(st.integers(min_value=1, max_value=3)))
        }
        stats.append(stat)
    return parts, stats

@given(parts_and_stats())
def test_apply_filters_empty_returns_all(data):
    parts, stats = data
    result_parts, result_stats = apply_filters(parts, stats, [])
    assert result_parts == parts
    assert result_stats == stats
```

**Failing input**: `parts=[], stats=[], filters=[]`

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import apply_filters

parts = []
stats = []
filters = []
apply_filters(parts, stats, filters)

parts = [{"piece": (0,)}]
stats = [{"columns": [{"name": "x", "min": 1, "max": 10}], "filter": False}]
filters = []
apply_filters(parts, stats, filters)
```

## Why This Is A Bug

The function's docstring does not prohibit passing an empty filters list, and it's reasonable to interpret an empty filters list as "no filtering" (i.e., return all parts). The current implementation crashes instead of handling this case gracefully. The expected behavior for an empty filter list would be to return all parts unchanged, similar to how many filtering operations work with empty filter sets.

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

The fix adds a check for an empty filters list before attempting to access `filters[0]`. When filters is empty, the function returns the input parts and statistics unchanged, which is the expected behavior for "no filtering".