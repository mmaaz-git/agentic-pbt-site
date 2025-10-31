# Bug Report: sorted_columns TypeError with None max Value

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with `TypeError: '>=' not supported between instances of 'int' and 'NoneType'` when the first row group has `None` values for min/max statistics and subsequent row groups have non-None values.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.dask_expr.io import sorted_columns


@st.composite
def statistics_with_none(draw):
    """Generate statistics where some row groups may have None min/max"""
    num_row_groups = draw(st.integers(min_value=2, max_value=5))
    column_name = draw(st.text(alphabet='abc', min_size=1, max_size=3))

    stats = []
    for i in range(num_row_groups):
        has_stats = draw(st.booleans())
        if has_stats:
            min_val = draw(st.integers(min_value=0, max_value=100))
            max_val = draw(st.integers(min_value=min_val, max_value=min_val + 10))
            col_stats = {"name": column_name, "min": min_val, "max": max_val}
        else:
            col_stats = {"name": column_name, "min": None, "max": None}

        stats.append({"columns": [col_stats]})

    return stats


@given(stats=statistics_with_none())
def test_sorted_columns_handles_none_gracefully(stats):
    """
    Property: sorted_columns should handle None min/max values without crashing.
    """
    result = sorted_columns(stats)

    assert isinstance(result, list)

    for col_info in result:
        divisions = col_info["divisions"]
        assert divisions == sorted(divisions)
```

**Failing input**:
```python
stats=[{'columns': [{'name': 'a', 'min': None, 'max': None}]},
       {'columns': [{'name': 'a', 'min': 0, 'max': 0}]}]
```

## Reproducing the Bug

```python
from dask.dataframe.dask_expr.io import sorted_columns

stats = [
    {"columns": [{"name": "a", "min": None, "max": None}]},
    {"columns": [{"name": "a", "min": 5, "max": 10}]},
]

result = sorted_columns(stats)
```

**Output**:
```
TypeError: '>=' not supported between instances of 'int' and 'NoneType'
```

## Why This Is A Bug

The function handles `None` values for `min` by checking `if c["min"] is None` before breaking out of the loop. However, it fails to check if `max` is `None` before performing the comparison `c["min"] >= max`. When the first row group has `max=None` and subsequent row groups have non-None `min` values, this comparison crashes.

This violates the function's contract of handling missing statistics gracefully. Row group statistics can legitimately have `None` values when statistics are not available for certain partitions.

## Fix

Check if `max` is `None` before comparison:

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -428,6 +428,10 @@ def sorted_columns(statistics, columns=None):
         for stats in statistics[1:]:
             c = stats["columns"][i]
             if c["min"] is None:
                 success = False
                 break
+            if max is None:
+                success = False
+                break
             if c["min"] >= max:
                 divisions.append(c["min"])
                 max = c["max"]
```

Or more concisely, check both at once:

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -428,6 +428,10 @@ def sorted_columns(statistics, columns=None):
         for stats in statistics[1:]:
             c = stats["columns"][i]
-            if c["min"] is None:
+            if c["min"] is None or max is None:
                 success = False
                 break
             if c["min"] >= max:
```