# Bug Report: dask.dataframe.io.parquet.core.sorted_columns TypeError with None Values

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with a TypeError when row-group statistics contain None values for max (but not min). The function attempts to sort divisions that include None values, causing Python's sort to fail.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.parquet.core import sorted_columns

@given(
    st.lists(
        st.dictionaries(
            st.just("columns"),
            st.lists(
                st.fixed_dictionaries({
                    "name": st.text(min_size=1, max_size=10),
                    "min": st.one_of(st.none(), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
                    "max": st.one_of(st.none(), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
                }),
                min_size=1,
                max_size=5
            ),
            min_size=1,
            max_size=1
        ),
        min_size=1,
        max_size=10
    )
)
def test_sorted_columns_divisions_are_sorted(statistics):
    result = sorted_columns(statistics)
    for item in result:
        divisions = item["divisions"]
        assert divisions == sorted(divisions)
```

**Failing input**: `statistics=[{'columns': [{'name': '0', 'min': 0, 'max': None}]}]`

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import sorted_columns

statistics = [{'columns': [{'name': '0', 'min': 0, 'max': None}]}]
result = sorted_columns(statistics)
```

Output:
```
TypeError: '<' not supported between instances of 'NoneType' and 'int'
```

## Why This Is A Bug

The function checks if `min` is not None (line 427) but fails to check if `max` is not None before:
1. Using it in comparison on line 433: `if c["min"] >= max`
2. Appending it to divisions on line 441: `divisions.append(max)`
3. Sorting divisions on line 442: `assert divisions == sorted(divisions)`

When max is None, the divisions list contains None values that cannot be compared with numeric values during sorting, causing a TypeError.

## Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -424,7 +424,7 @@ def sorted_columns(statistics, columns=None):
             continue
         divisions = [c["min"]]
         max = c["max"]
-        success = c["min"] is not None
+        success = c["min"] is not None and c["max"] is not None
         for stats in statistics[1:]:
             c = stats["columns"][i]
             if c["min"] is None:
@@ -433,6 +433,10 @@ def sorted_columns(statistics, columns=None):
             if c["min"] >= max:
                 divisions.append(c["min"])
                 max = c["max"]
+                if max is None:
+                    success = False
+                    break
             else:
                 success = False
                 break
```