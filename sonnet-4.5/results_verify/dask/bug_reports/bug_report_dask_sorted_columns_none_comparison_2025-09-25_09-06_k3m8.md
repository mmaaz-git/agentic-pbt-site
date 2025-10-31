# Bug Report: dask.dataframe.io.parquet.core.sorted_columns None Comparison Crash

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with a TypeError when comparing `None` max values with integer min values during statistics processing.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.parquet.core import sorted_columns
import string


@st.composite
def statistics_strategy(draw):
    num_row_groups = draw(st.integers(min_value=0, max_value=20))
    num_columns = draw(st.integers(min_value=1, max_value=5))

    column_names = [
        draw(st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=10))
        for _ in range(num_columns)
    ]

    stats = []
    for _ in range(num_row_groups):
        columns = []
        for col_name in column_names:
            has_stats = draw(st.booleans())
            if has_stats:
                min_val = draw(st.integers(min_value=-1000, max_value=1000) | st.none())
                if min_val is not None:
                    max_val = draw(
                        st.integers(min_value=min_val, max_value=1000) | st.none()
                    )
                else:
                    max_val = None
                columns.append({"name": col_name, "min": min_val, "max": max_val})
            else:
                columns.append({"name": col_name})

        stats.append({"columns": columns})

    return stats


@given(statistics_strategy())
@settings(max_examples=1000)
def test_sorted_columns_divisions_are_sorted(statistics):
    result = sorted_columns(statistics)
    for col_info in result:
        divisions = col_info["divisions"]
        assert divisions == sorted(divisions)
```

**Failing input**: `statistics=[{'columns': [{'name': 'a'}, {'name': 'a', 'min': None, 'max': None}]}, {'columns': [{'name': 'a'}, {'name': 'a', 'min': 0, 'max': None}]}]`

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import sorted_columns

statistics = [
    {"columns": [{"name": "col", "min": 1, "max": None}]},
    {"columns": [{"name": "col", "min": 2, "max": 3}]},
]

result = sorted_columns(statistics)
```

## Why This Is A Bug

Parquet statistics can legitimately have `None` for max values when statistics are incomplete or unavailable. The function should handle this case gracefully instead of crashing with a TypeError when comparing `None` with integers.

## Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -430,7 +430,7 @@ def sorted_columns(statistics, columns=None):
             if c["min"] is None:
                 success = False
                 break
-            if c["min"] >= max:
+            if max is not None and c["min"] >= max:
                 divisions.append(c["min"])
                 max = c["max"]
             else:
```