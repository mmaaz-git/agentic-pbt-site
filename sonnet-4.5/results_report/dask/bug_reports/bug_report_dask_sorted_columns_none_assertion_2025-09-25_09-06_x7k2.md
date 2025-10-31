# Bug Report: dask.dataframe.io.parquet.core.sorted_columns Assertion Failure with None

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with a TypeError in its internal assertion when divisions contain `None` values, as `None` cannot be compared with integers during sorting.

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
                        st.integers(min_value=min_value, max_value=1000) | st.none()
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

**Failing input**: `statistics=[{'columns': [{'name': 'a', 'min': 0, 'max': None}]}]`

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import sorted_columns

statistics = [{"columns": [{"name": "a", "min": 0, "max": None}]}]

result = sorted_columns(statistics)
```

## Why This Is A Bug

When a column has statistics with a `None` max value, the function adds this `None` to the divisions list. The internal assertion `assert divisions == sorted(divisions)` then fails because Python cannot compare `None` with integers. This can occur with legitimate parquet files that have incomplete statistics.

## Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -424,6 +424,8 @@ def sorted_columns(statistics, columns=None):
         divisions = [c["min"]]
         max = c["max"]
         success = c["min"] is not None
+        if max is None:
+            success = False
         for stats in statistics[1:]:
             c = stats["columns"][i]
             if c["min"] is None:
```