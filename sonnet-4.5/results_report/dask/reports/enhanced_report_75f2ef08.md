# Bug Report: dask.dataframe.io.parquet.core.sorted_columns TypeError with None max values

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with a TypeError when parquet row-group statistics contain None values for max (but not min), causing Python's sort operation to fail when comparing None with numeric values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from dask.dataframe.io.parquet.core import sorted_columns
import traceback

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
@example(statistics=[{'columns': [{'name': '0', 'min': 0, 'max': None}]}])  # The failing example
@settings(max_examples=100)
def test_sorted_columns_divisions_are_sorted(statistics):
    """Test that sorted_columns returns properly sorted divisions without crashing"""
    try:
        result = sorted_columns(statistics)
        # If we get a result, verify the divisions are sorted
        for item in result:
            divisions = item["divisions"]
            # Check that divisions are sorted (this should always be true per the assertion in the function)
            assert divisions == sorted(divisions), f"Divisions not sorted: {divisions}"
    except TypeError as e:
        # If we hit the bug, print details
        print(f"\nFailing input: {statistics}")
        print(f"TypeError: {e}")
        traceback.print_exc()
        raise

# Run the test
if __name__ == "__main__":
    print("Running property-based test for sorted_columns...")
    try:
        test_sorted_columns_divisions_are_sorted()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
```

<details>

<summary>
**Failing input**: `[{'columns': [{'name': '0', 'min': 0, 'max': None}]}]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 30, in test_sorted_columns_divisions_are_sorted
    result = sorted_columns(statistics)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 442, in sorted_columns
    assert divisions == sorted(divisions)
                        ~~~~~~^^^^^^^^^^^
TypeError: '<' not supported between instances of 'NoneType' and 'int'
Running property-based test for sorted_columns...

Failing input: [{'columns': [{'name': '0', 'min': 0, 'max': None}]}]
TypeError: '<' not supported between instances of 'NoneType' and 'int'
Test failed with error: '<' not supported between instances of 'NoneType' and 'int'
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import sorted_columns

# Test case from the bug report - min is valid but max is None
statistics = [{'columns': [{'name': '0', 'min': 0, 'max': None}]}]

try:
    result = sorted_columns(statistics)
    print(f"Result: {result}")
except TypeError as e:
    print(f"TypeError: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
TypeError when sorting divisions containing None
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/repo.py", line 7, in <module>
    result = sorted_columns(statistics)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 442, in sorted_columns
    assert divisions == sorted(divisions)
                        ~~~~~~^^^^^^^^^^^
TypeError: '<' not supported between instances of 'NoneType' and 'int'
TypeError: '<' not supported between instances of 'NoneType' and 'int'
```
</details>

## Why This Is A Bug

The `sorted_columns` function is designed to identify sorted columns in parquet row-group statistics and return appropriate divisions for optimization. The function's behavior with None values is inconsistent:

1. **Existing None handling**: The function already handles None values for the `min` statistic (lines 427, 430-432), setting `success = False` and skipping columns with None min values.

2. **Missing None handling for max**: The function fails to check if `max` is None before:
   - Using it in comparison on line 433: `if c["min"] >= max` (comparing with None fails)
   - Appending it to divisions on line 441: `divisions.append(max)`
   - Sorting divisions on line 442: `assert divisions == sorted(divisions)` (sorting fails with TypeError)

3. **Legitimate scenarios for None statistics**: Parquet files can have None statistics when:
   - Statistics collection was disabled during write
   - Columns contain only NULL values
   - Legacy parquet files with incomplete statistics
   - Certain data types that don't support min/max statistics

4. **Expected behavior**: Based on how None min values are handled, the function should gracefully skip columns with None max values rather than crashing. The function successfully returns an empty list when both min and max are None, demonstrating it's designed to handle missing statistics.

## Relevant Context

The `sorted_columns` function is located in `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py` starting at line 402.

This is an internal utility function used during parquet file reading to optimize queries by identifying pre-sorted columns. While not part of the public API, it affects users reading parquet files with Dask when those files have partial statistics.

Documentation link: The function has a docstring but is not documented in Dask's main documentation as it's an internal utility.

Code location: [dask/dataframe/io/parquet/core.py:402-445](https://github.com/dask/dask/blob/main/dask/dataframe/io/parquet/core.py)

## Proposed Fix

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