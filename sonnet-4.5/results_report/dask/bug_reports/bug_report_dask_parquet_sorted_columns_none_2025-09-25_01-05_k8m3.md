# Bug Report: dask.dataframe.io.parquet sorted_columns TypeError with None Values

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with a TypeError when parquet statistics contain None values for min or max fields. This occurs because the function checks for None in the min field but fails to validate None in the max field before performing comparisons and sorting operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@st.composite
def statistics_with_nones(draw):
    num_row_groups = draw(st.integers(min_value=1, max_value=10))
    col_name = "test_col"

    statistics = []
    for i in range(num_row_groups):
        has_min = draw(st.booleans())
        has_max = draw(st.booleans())

        min_val = draw(st.integers(min_value=-100, max_value=100)) if has_min else None
        max_val = draw(st.integers(min_value=-100, max_value=100)) if has_max else None

        if min_val is not None and max_val is not None and min_val > max_val:
            min_val, max_val = max_val, min_val

        statistics.append({
            "columns": [{
                "name": col_name,
                "min": min_val,
                "max": max_val
            }]
        })

    return statistics, col_name

@given(statistics_with_nones())
@settings(max_examples=500)
def test_sorted_columns_none_handling(data):
    statistics, col_name = data
    result = sorted_columns(statistics, columns=[col_name])

    for item in result:
        divisions = item["divisions"]
        assert None not in divisions
        assert divisions == sorted(divisions)
```

**Failing input**:
```python
[{'columns': [{'name': 'test_col', 'min': None, 'max': None}]},
 {'columns': [{'name': 'test_col', 'min': 0, 'max': None}]}]
```

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import sorted_columns

statistics = [
    {'columns': [{'name': 'test_col', 'min': None, 'max': None}]},
    {'columns': [{'name': 'test_col', 'min': 0, 'max': None}]}
]

sorted_columns(statistics, columns=['test_col'])
```

**Output:**
```
TypeError: '>=' not supported between instances of 'int' and 'NoneType'
```

**Alternative failing case:**
```python
statistics = [
    {'columns': [{'name': 'test_col', 'min': 0, 'max': None}]}
]

sorted_columns(statistics, columns=['test_col'])
```

**Output:**
```
TypeError: '<' not supported between instances of 'NoneType' and 'int'
```

## Why This Is A Bug

The `sorted_columns` function is designed to work with parquet row-group statistics which can legitimately have None values when statistics are missing or incomplete. The function already attempts to handle None values (see line 427, 430 in core.py) but has incomplete validation:

1. **Line 426**: `max = c["max"]` sets max without checking if it's None
2. **Line 433**: `if c["min"] >= max:` crashes when max is None
3. **Line 441**: `divisions.append(max)` adds None to divisions list
4. **Line 442**: `assert divisions == sorted(divisions)` crashes when trying to sort a list containing None

This affects real users when reading parquet files with incomplete or missing statistics metadata.

## Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -423,13 +423,18 @@ def sorted_columns(statistics, columns=None):
             continue
         divisions = [c["min"]]
         max = c["max"]
-        success = c["min"] is not None
+        # Check both min and max are not None for the first row group
+        success = c["min"] is not None and c["max"] is not None
         for stats in statistics[1:]:
             c = stats["columns"][i]
-            if c["min"] is None:
+            # Check both min and max are not None
+            if c["min"] is None or c["max"] is None:
                 success = False
                 break
+            # Check max from previous iteration is not None
+            if max is None:
+                success = False
+                break
             if c["min"] >= max:
                 divisions.append(c["min"])
                 max = c["max"]
```