# Bug Report: dask.dataframe.io.parquet sorted_columns AssertionError with Invalid Statistics

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with an `AssertionError` when given statistics where `min > max` in a column, rather than gracefully handling or rejecting the invalid data.

## Property-Based Test

```python
from hypothesis import given, assume, settings, strategies as st
import dask.dataframe.io.parquet.core as core

@given(st.lists(
    st.lists(
        st.dictionaries(
            st.sampled_from(["name", "min", "max"]),
            st.one_of(
                st.text(min_size=1),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False)
            )
        ),
        min_size=1,
        max_size=5
    ),
    min_size=1,
    max_size=10
))
def test_sorted_columns_divisions_sorted(raw_stats):
    statistics = []
    for row_group in raw_stats:
        stat_dict = {"columns": []}
        for col_dict in row_group:
            if "name" in col_dict:
                stat_dict["columns"].append(col_dict)
        if stat_dict["columns"]:
            statistics.append(stat_dict)

    if not statistics:
        assume(False)

    result = core.sorted_columns(statistics)

    for col_info in result:
        divisions = col_info['divisions']
        assert divisions == sorted(divisions)
```

**Failing input**: `statistics=[{"columns": [{"name": "0", "min": "00", "max": "0"}]}]`

## Reproducing the Bug

```python
import dask.dataframe.io.parquet.core as core

statistics = [
    {
        "columns": [
            {"name": "0", "min": "00", "max": "0"}
        ]
    }
]

result = core.sorted_columns(statistics)
```

This raises:
```
AssertionError
```

The assertion that fails is at line 442 of `core.py`:
```python
assert divisions == sorted(divisions)
```

In this case, `divisions = ['00', '0']` but `sorted(divisions) = ['0', '00']`.

## Why This Is A Bug

The function builds divisions assuming that if it constructs them by checking `min >= max` relationships between row groups, they will be sorted. However, it doesn't validate that within each row group, `min <= max`. When `min > max` (as in the test case where '00' > '0' lexicographically), the function:

1. Adds the min value to divisions: `['00']`
2. Adds the max value to divisions: `['00', '0']`
3. Asserts that divisions are sorted
4. The assertion fails because `sorted(['00', '0']) = ['0', '00']`

While it's true that parquet statistics with `min > max` represent invalid/corrupted data, the function should handle this gracefully (either by skipping such columns or raising a meaningful exception) rather than hitting an assertion error.

## Fix

Add validation to skip columns where min > max in any row group:

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -423,6 +423,10 @@ def sorted_columns(statistics, columns=None):
         if not all(
             "min" in s["columns"][i] and "max" in s["columns"][i] for s in statistics
         ):
             continue
+        if not all(
+            s["columns"][i]["min"] <= s["columns"][i]["max"] for s in statistics
+        ):
+            continue
         divisions = [c["min"]]
         max = c["max"]
         success = c["min"] is not None
```