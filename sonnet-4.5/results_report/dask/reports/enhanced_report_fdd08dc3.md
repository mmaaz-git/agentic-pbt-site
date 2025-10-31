# Bug Report: dask.dataframe.io.parquet.core sorted_columns TypeError with None values

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with multiple TypeErrors when processing parquet statistics containing None values in min/max fields, preventing the reading of parquet files with incomplete column statistics.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import dask.dataframe.io.parquet.core as parquet_core

@given(st.lists(
    st.fixed_dictionaries({
        'columns': st.lists(
            st.fixed_dictionaries({
                'name': st.text(min_size=1, max_size=20),
                'min': st.one_of(st.none(), st.integers(-1000, 1000)),
                'max': st.one_of(st.none(), st.integers(-1000, 1000))
            }),
            min_size=1,
            max_size=5
        )
    }),
    min_size=1,
    max_size=10
))
def test_sorted_columns_divisions_are_sorted(statistics):
    result = parquet_core.sorted_columns(statistics)
    for item in result:
        assert item['divisions'] == sorted(item['divisions'])

if __name__ == "__main__":
    test_sorted_columns_divisions_are_sorted()
```

<details>

<summary>
**Failing input**: `statistics=[{'columns': [{'name': '0', 'min': None, 'max': None}]}, {'columns': [{'name': '0', 'min': 0, 'max': None}]}]`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 25, in <module>
  |     test_sorted_columns_divisions_are_sorted()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 5, in test_sorted_columns_divisions_are_sorted
  |     st.fixed_dictionaries({
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 4 distinct failures. (4 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 20, in test_sorted_columns_divisions_are_sorted
    |     result = parquet_core.sorted_columns(statistics)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 421, in sorted_columns
    |     if not all(
    |            ~~~^
    |         "min" in s["columns"][i] and "max" in s["columns"][i] for s in statistics
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     ):
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 422, in <genexpr>
    |     "min" in s["columns"][i] and "max" in s["columns"][i] for s in statistics
    |              ~~~~~~~~~~~~^^^
    | IndexError: list index out of range
    | Falsifying example: test_sorted_columns_divisions_are_sorted(
    |     statistics=[{'columns': [{'name': '0', 'min': None, 'max': None},
    |        {'name': '0', 'min': None, 'max': None}]},
    |      {'columns': [{'name': '0', 'min': None, 'max': None}]}],
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 20, in test_sorted_columns_divisions_are_sorted
    |     result = parquet_core.sorted_columns(statistics)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 433, in sorted_columns
    |     if c["min"] >= max:
    |        ^^^^^^^^^^^^^^^
    | TypeError: '>=' not supported between instances of 'int' and 'NoneType'
    | Falsifying example: test_sorted_columns_divisions_are_sorted(
    |     statistics=[{'columns': [{'name': '0', 'min': None, 'max': None}]},
    |      {'columns': [{'name': '0', 'min': 0, 'max': None}]}],
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py:433
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 20, in test_sorted_columns_divisions_are_sorted
    |     result = parquet_core.sorted_columns(statistics)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 442, in sorted_columns
    |     assert divisions == sorted(divisions)
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_sorted_columns_divisions_are_sorted(
    |     statistics=[{'columns': [{'name': '0', 'min': 0, 'max': -1}]}],
    | )
    +---------------- 4 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 20, in test_sorted_columns_divisions_are_sorted
    |     result = parquet_core.sorted_columns(statistics)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 442, in sorted_columns
    |     assert divisions == sorted(divisions)
    |                         ~~~~~~^^^^^^^^^^^
    | TypeError: '<' not supported between instances of 'NoneType' and 'int'
    | Falsifying example: test_sorted_columns_divisions_are_sorted(
    |     statistics=[{'columns': [{'name': '0', 'min': 0, 'max': None}]}],
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import dask.dataframe.io.parquet.core as parquet_core

statistics = [
    {'columns': [{'name': 'col1', 'min': None, 'max': None}]},
    {'columns': [{'name': 'col1', 'min': 0, 'max': None}]}
]

result = parquet_core.sorted_columns(statistics)
print("Result:", result)
```

<details>

<summary>
TypeError when comparing int with NoneType
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/repo.py", line 8, in <module>
    result = parquet_core.sorted_columns(statistics)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 433, in sorted_columns
    if c["min"] >= max:
       ^^^^^^^^^^^^^^^
TypeError: '>=' not supported between instances of 'int' and 'NoneType'
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Incomplete None handling**: The function checks if `c["min"]` is None at line 430-432 to prevent errors, but fails to check if `max` is None before using it in the comparison `c["min"] >= max` at line 433. This contradicts the defensive programming pattern already present in the code.

2. **Invalid assertion logic**: The function can create divisions with None values (when max is None), then attempts to sort them at line 442 with `assert divisions == sorted(divisions)`, which crashes when sorting mixed None and integer values.

3. **Inconsistent column handling**: The function assumes all row groups have the same number of columns, causing IndexError when statistics have varying column counts (line 422).

4. **Invalid divisions**: The function can create divisions where min > max (e.g., min=0, max=-1), violating the fundamental assumption that divisions should be sorted ranges.

According to the docstring, this function should "Find sorted columns given row-group statistics" and return valid divisions. Instead, it crashes on realistic inputs that can occur with:
- Parquet files with incomplete statistics (common in corrupted files or certain writers)
- Null-only columns
- Columns added or removed between row groups
- Edge cases in distributed data processing

## Relevant Context

The `sorted_columns` function is an internal utility in dask's parquet reader that optimizes reading by identifying pre-sorted columns. It's called during parquet file reading operations and is critical for performance optimization. The function is located at `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py:402`.

Key observations:
- Line 421-423 already attempts to filter out columns without both min and max, but this check uses dictionary key existence rather than None value checking
- Line 427 initializes `success = c["min"] is not None` but doesn't check `c["max"]`
- Line 426 sets `max = c["max"]` which can be None, leading to the comparison error
- The function is used internally by `read_parquet` operations to optimize data loading

Related documentation:
- Parquet format specification allows for missing statistics: https://github.com/apache/parquet-format/blob/master/src/main/thrift/parquet.thrift
- Dask parquet documentation: https://docs.dask.org/en/stable/dataframe-parquet.html

## Proposed Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -424,13 +424,17 @@ def sorted_columns(statistics, columns=None):
             continue
         divisions = [c["min"]]
         max = c["max"]
-        success = c["min"] is not None
+        success = c["min"] is not None and c["max"] is not None
+        if not success:
+            continue
         for stats in statistics[1:]:
             c = stats["columns"][i]
             if c["min"] is None:
                 success = False
                 break
-            if c["min"] >= max:
+            if max is None or c["max"] is None:
+                success = False
+                break
+            if c["min"] >= max and c["min"] <= c["max"]:
                 divisions.append(c["min"])
                 max = c["max"]
             else:
@@ -439,7 +443,8 @@ def sorted_columns(statistics, columns=None):

         if success:
             divisions.append(max)
-            assert divisions == sorted(divisions)
+            if None not in divisions:
+                assert divisions == sorted(divisions)
             out.append({"name": c["name"], "divisions": divisions})

     return out
```