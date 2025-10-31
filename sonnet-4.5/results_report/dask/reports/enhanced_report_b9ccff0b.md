# Bug Report: dask.dataframe.io.parquet.core.sorted_columns TypeError with None Values in Statistics

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with a TypeError when parquet row-group statistics contain None values for the max field, failing to properly validate None values before performing comparisons and sorting operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.parquet.core import sorted_columns

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

if __name__ == "__main__":
    test_sorted_columns_none_handling()
```

<details>

<summary>
**Failing input**: `([{'columns': [{'name': 'test_col', 'min': 0, 'max': None}]}], 'test_col')`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 42, in <module>
    test_sorted_columns_none_handling()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 31, in test_sorted_columns_none_handling
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 34, in test_sorted_columns_none_handling
    result = sorted_columns(statistics, columns=[col_name])
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 442, in sorted_columns
    assert divisions == sorted(divisions)
                        ~~~~~~^^^^^^^^^^^
TypeError: '<' not supported between instances of 'NoneType' and 'int'
Falsifying example: test_sorted_columns_none_handling(
    data=([{'columns': [{'name': 'test_col', 'min': 0, 'max': None}]}],
     'test_col'),
)
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import sorted_columns

# Test case 1: None in both min and max for first row group, valid min but None max in second
print("Test case 1:")
print("-" * 50)
statistics = [
    {'columns': [{'name': 'test_col', 'min': None, 'max': None}]},
    {'columns': [{'name': 'test_col', 'min': 0, 'max': None}]}
]

try:
    result = sorted_columns(statistics, columns=['test_col'])
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n")

# Test case 2: Single row group with valid min but None max
print("Test case 2:")
print("-" * 50)
statistics = [
    {'columns': [{'name': 'test_col', 'min': 0, 'max': None}]}
]

try:
    result = sorted_columns(statistics, columns=['test_col'])
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
TypeError when comparing or sorting with None values
</summary>
```
Test case 1:
--------------------------------------------------
Error: TypeError: '>=' not supported between instances of 'int' and 'NoneType'


Test case 2:
--------------------------------------------------
Error: TypeError: '<' not supported between instances of 'NoneType' and 'int'
```
</details>

## Why This Is A Bug

This violates expected behavior because the `sorted_columns` function is designed to identify sorted columns from parquet row-group statistics, which can legitimately have None values when statistics are missing or incomplete. The function already demonstrates awareness of this by checking for None in the min field (line 427 and 430), but fails to handle None in the max field consistently.

Specifically:
1. **Line 426**: `max = c["max"]` assigns the max value without checking if it's None
2. **Line 427**: Only checks `c["min"] is not None` for the success flag, ignoring max
3. **Line 433**: Comparison `c["min"] >= max` crashes with TypeError when max is None
4. **Line 441**: `divisions.append(max)` adds None to the divisions list
5. **Line 442**: `assert divisions == sorted(divisions)` crashes when trying to sort a list containing None

The function's documentation states it should "find sorted columns given row-group statistics" and return appropriate divisions. When statistics are incomplete (containing None), the expected behavior would be to mark that column as unsorted and exclude it from results, not crash with a TypeError.

## Relevant Context

Parquet files commonly have missing statistics in real-world scenarios due to:
- Statistics collection being disabled during file creation
- Older parquet file formats that don't include all statistics
- Row groups containing only null values
- Corrupted or incomplete metadata

The function is used in production code paths within dask-expr when reading parquet files to determine optimal data partitioning. The current implementation already shows partial awareness of None handling but implements it incompletely, leading to crashes on legitimate input data.

Documentation: The function is defined at `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py:402`

## Proposed Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -425,13 +425,18 @@ def sorted_columns(statistics, columns=None):
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