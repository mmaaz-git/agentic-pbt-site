# Bug Report: dask.dataframe.io.parquet.core.sorted_columns TypeError with None Max Values

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with a TypeError when attempting to sort a list containing both integers and None values, which occurs when parquet statistics have a valid min value but a None max value.

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

if __name__ == "__main__":
    test_sorted_columns_divisions_are_sorted()
```

<details>

<summary>
**Failing input**: `statistics=[{'columns': [{'name': 'a', 'min': 0, 'max': None}]}]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 47, in <module>
    test_sorted_columns_divisions_are_sorted()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 39, in test_sorted_columns_divisions_are_sorted
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 41, in test_sorted_columns_divisions_are_sorted
    result = sorted_columns(statistics)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 442, in sorted_columns
    assert divisions == sorted(divisions)
                        ~~~~~~^^^^^^^^^^^
TypeError: '<' not supported between instances of 'NoneType' and 'int'
Falsifying example: test_sorted_columns_divisions_are_sorted(
    statistics=[{'columns': [{'name': 'a', 'min': 0, 'max': None}]}],
)
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import sorted_columns

# Minimal reproduction of the bug
statistics = [{"columns": [{"name": "a", "min": 0, "max": None}]}]

print("Input statistics:")
print(f"  statistics = {statistics}")
print()
print("Calling sorted_columns(statistics)...")
print()

try:
    result = sorted_columns(statistics)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    print()
    print("Full traceback:")
    traceback.print_exc()
```

<details>

<summary>
TypeError when sorting list containing None and integer
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/repo.py", line 13, in <module>
    result = sorted_columns(statistics)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 442, in sorted_columns
    assert divisions == sorted(divisions)
                        ~~~~~~^^^^^^^^^^^
TypeError: '<' not supported between instances of 'NoneType' and 'int'
Input statistics:
  statistics = [{'columns': [{'name': 'a', 'min': 0, 'max': None}]}]

Calling sorted_columns(statistics)...

Error occurred: TypeError: '<' not supported between instances of 'NoneType' and 'int'

Full traceback:
```
</details>

## Why This Is A Bug

This violates expected behavior because the function is designed to handle incomplete statistics but fails to do so consistently. The code explicitly handles None values for the `min` field (lines 427 and 430-432) but neglects to handle None values for the `max` field before appending to the divisions list. When a column has `min=0` and `max=None`, the function:

1. Passes the initial check at lines 421-424 (which only verifies that "min" and "max" keys exist, not that their values are non-None)
2. Sets `success = True` at line 427 because `c["min"] is not None`
3. Appends the None max value to divisions at line 441
4. Crashes at line 442 when trying to sort a list containing both integers and None

This contradicts the function's partial handling of None values and causes an unhandled crash rather than gracefully returning no sorted columns for incomplete statistics.

## Relevant Context

The `sorted_columns` function is used internally by dask's parquet reader to optimize column access by identifying pre-sorted columns in parquet files. Parquet files can have incomplete statistics when:
- Files are written with certain writers that don't compute all statistics
- Statistics are corrupted or partially available
- Files use older parquet formats with limited statistics support

The function's docstring indicates it "finds all columns that are sorted" and returns a list of dictionaries with column names and their divisions. The current implementation assumes that if statistics exist (the keys are present), the values are either all None or all valid, but doesn't account for mixed cases.

Relevant code location: `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py:402-445`

## Proposed Fix

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