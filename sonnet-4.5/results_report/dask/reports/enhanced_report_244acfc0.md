# Bug Report: dask.dataframe.io.parquet.core.sorted_columns TypeError with None Statistics

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with a TypeError when comparing an integer to None, occurring when the first row group has None min/max statistics and subsequent row groups have valid integer statistics.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.parquet.core import sorted_columns


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
@settings(max_examples=100)
def test_sorted_columns_handles_none_gracefully(stats):
    """
    Property: sorted_columns should handle None min/max values without crashing.
    """
    result = sorted_columns(stats)

    assert isinstance(result, list)

    for col_info in result:
        divisions = col_info["divisions"]
        assert divisions == sorted(divisions)


if __name__ == "__main__":
    # Run the test
    test_sorted_columns_handles_none_gracefully()
```

<details>

<summary>
**Failing input**: `stats=[{'columns': [{'name': 'a', 'min': None, 'max': None}]}, {'columns': [{'name': 'a', 'min': 0, 'max': 0}]}]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 43, in <module>
    test_sorted_columns_handles_none_gracefully()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 27, in test_sorted_columns_handles_none_gracefully
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 32, in test_sorted_columns_handles_none_gracefully
    result = sorted_columns(stats)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 433, in sorted_columns
    if c["min"] >= max:
       ^^^^^^^^^^^^^^^
TypeError: '>=' not supported between instances of 'int' and 'NoneType'
Falsifying example: test_sorted_columns_handles_none_gracefully(
    stats=[{'columns': [{'name': 'a', 'min': None, 'max': None}]},
     {'columns': [{'name': 'a', 'min': 0, 'max': 0}]}],
)
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import sorted_columns

stats = [
    {"columns": [{"name": "a", "min": None, "max": None}]},
    {"columns": [{"name": "a", "min": 5, "max": 10}]},
]

result = sorted_columns(stats)
print("Result:", result)
```

<details>

<summary>
TypeError when comparing int to NoneType
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/repo.py", line 8, in <module>
    result = sorted_columns(stats)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 433, in sorted_columns
    if c["min"] >= max:
       ^^^^^^^^^^^^^^^
TypeError: '>=' not supported between instances of 'int' and 'NoneType'
```
</details>

## Why This Is A Bug

This violates the expected behavior of gracefully handling missing statistics in Parquet row groups. The function demonstrates clear intent to handle None values through existing checks at lines 427 and 430-432, but fails to check if `max` is None before performing the comparison at line 433.

The function already implements defensive programming by checking `if c["min"] is None` (line 430) to avoid crashes, and initializes `success = c["min"] is not None` (line 427) to handle the first row group's None case. However, when the first row group has `max=None` and a subsequent row group has a non-None `min`, the comparison `c["min"] >= max` crashes because Python cannot compare an integer with None.

This is particularly problematic because Parquet files can legitimately have missing statistics when statistics collection is disabled, for certain data types, or when dealing with legacy files. The inconsistent None handling makes the function unreliable for real-world Parquet files where statistics may be partially available.

## Relevant Context

The `sorted_columns` function is used to identify columns in Parquet files that are sorted across row groups, which enables optimizations in query execution. The function examines row group statistics (min/max values) to determine if columns maintain sort order.

Key code locations:
- Function definition: `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py:402`
- Existing None check for `c["min"]`: line 430
- Missing None check for `max`: line 433

The function already handles several edge cases:
1. Empty statistics list (line 414-415)
2. Missing min/max keys in statistics (lines 421-424)
3. None values for `c["min"]` in subsequent row groups (lines 430-432)

But misses the case where `max` from the first row group is None.

## Proposed Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -428,7 +428,7 @@ def sorted_columns(statistics, columns=None):
         for stats in statistics[1:]:
             c = stats["columns"][i]
-            if c["min"] is None:
+            if c["min"] is None or max is None:
                 success = False
                 break
             if c["min"] >= max:
```