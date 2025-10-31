# Bug Report: dask.dataframe.io.parquet apply_filters TypeError with Partial None Statistics

**Target**: `dask.dataframe.io.parquet.core.apply_filters`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `apply_filters` function crashes with a TypeError when parquet row-group statistics have only one of min or max as None, causing comparison operations between None and integers to fail during filter evaluation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.parquet.core import apply_filters

@st.composite
def filter_with_null_count_data(draw):
    num_parts = draw(st.integers(min_value=1, max_value=10))
    col_name = "x"

    parts = []
    statistics = []

    for i in range(num_parts):
        min_val = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=100)))
        max_val = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=100)))

        if min_val is not None and max_val is not None and min_val > max_val:
            min_val, max_val = max_val, min_val

        null_count = draw(st.integers(min_value=0, max_value=100))

        parts.append({"id": i})
        statistics.append({
            "filter": False,
            "columns": [{
                "name": col_name,
                "min": min_val,
                "max": max_val,
                "null_count": null_count
            }]
        })

    return parts, statistics, col_name

@given(filter_with_null_count_data())
@settings(max_examples=300)
def test_apply_filters_with_nulls_no_crash(data):
    parts, statistics, col_name = data

    filtered_parts, filtered_stats = apply_filters(
        parts, statistics, [(col_name, "=", 50)]
    )
    assert len(filtered_parts) <= len(parts)

if __name__ == "__main__":
    test_apply_filters_with_nulls_no_crash()
```

<details>

<summary>
**Failing input**: `([{'id': 0}], [{'filter': False, 'columns': [{'name': 'x', 'min': None, 'max': 0, 'null_count': 0}]}], 'x')`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 45, in <module>
    test_apply_filters_with_nulls_no_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 35, in test_apply_filters_with_nulls_no_crash
    @settings(max_examples=300)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 39, in test_apply_filters_with_nulls_no_crash
    filtered_parts, filtered_stats = apply_filters(
                                     ~~~~~~~~~~~~~^
        parts, statistics, [(col_name, "=", 50)]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 558, in apply_filters
    out_parts, out_statistics = apply_conjunction(parts, statistics, conjunction)
                                ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 533, in apply_conjunction
    and min <= value <= max
        ^^^^^^^^^^^^^^^^^^^
TypeError: '<=' not supported between instances of 'NoneType' and 'int'
Falsifying example: test_apply_filters_with_nulls_no_crash(
    data=([{'id': 0}],
     [{'filter': False,
       'columns': [{'name': 'x', 'min': None, 'max': 0, 'null_count': 0}]}],
     'x'),
)
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import apply_filters

# Case 1: min is None, max has a value
parts = [{'id': 0}]
statistics = [{
    'filter': False,
    'columns': [{
        'name': 'x',
        'min': None,
        'max': 0,
        'null_count': 0
    }]
}]

print("Test case 1: min=None, max=0")
print("Input parts:", parts)
print("Input statistics:", statistics)
print("Filter: [('x', '=', 50)]")

try:
    result = apply_filters(parts, statistics, [('x', '=', 50)])
    print("Result:", result)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*60 + "\n")

# Case 2: max is None, min has a value
parts2 = [{'id': 0}]
statistics2 = [{
    'filter': False,
    'columns': [{
        'name': 'x',
        'min': 0,
        'max': None,
        'null_count': 0
    }]
}]

print("Test case 2: min=0, max=None")
print("Input parts:", parts2)
print("Input statistics:", statistics2)
print("Filter: [('x', '=', 50)]")

try:
    result2 = apply_filters(parts2, statistics2, [('x', '=', 50)])
    print("Result:", result2)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
TypeError when comparing None with integers
</summary>
```
Test case 1: min=None, max=0
Input parts: [{'id': 0}]
Input statistics: [{'filter': False, 'columns': [{'name': 'x', 'min': None, 'max': 0, 'null_count': 0}]}]
Filter: [('x', '=', 50)]
Error: TypeError: '<=' not supported between instances of 'NoneType' and 'int'

============================================================

Test case 2: min=0, max=None
Input parts: [{'id': 0}]
Input statistics: [{'filter': False, 'columns': [{'name': 'x', 'min': 0, 'max': None, 'null_count': 0}]}]
Filter: [('x', '=', 50)]
Error: TypeError: '<=' not supported between instances of 'int' and 'NoneType'
```
</details>

## Why This Is A Bug

The Apache Parquet format specification explicitly allows column statistics to have partial None values where either min or max (but not necessarily both) can be None. This is valid metadata that readers must handle gracefully. The `apply_filters` function is designed to handle partition-level filtering using these statistics, but fails to properly validate for partial None cases before performing comparison operations.

The code contains a comment on line 531 stating "(main/max cannot be None for remaining checks)" which acknowledges this requirement, but the actual implementation doesn't enforce this invariant. The existing None checks (lines 519-529) only handle specific cases:
- Both min and max are None with no null_count (line 519)
- Both min and max are None with null_count (lines 526-529)
- Special handling for "is" and "is not" operators (lines 521-524)

However, these checks fail to catch the case where exactly one of min or max is None, allowing execution to fall through to the comparison operators (lines 532-547) which crash when attempting to compare None with numeric values.

## Relevant Context

This bug affects real-world Parquet files in several scenarios:
- Files created by older Parquet writers that may omit certain statistics
- Files with incomplete or corrupted metadata
- Files where statistics collection was disabled for specific columns
- Legacy Parquet files migrated from other storage formats

The issue is similar to previously reported bug #9764 in the Dask codebase, which dealt with None statistics causing TypeErrors. This indicates that handling partial statistics is a known challenge in the codebase.

The function's docstring states it performs "partition-level (hive) filtering" to prevent loading unnecessary row-groups, which is a critical performance optimization for large datasets. When this crashes, users lose the ability to efficiently filter large Parquet datasets.

Source code location: `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py:448-565`

## Proposed Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -516,6 +516,10 @@ def apply_filters(parts, statistics, filters):
                 else:
                     if (
                         # Must allow row-groups with "missing" stats
+                        # Skip if either min or max is None (partial statistics)
+                        # for comparison operators that require both values
+                        (min is None or max is None)
+                        and operator in ("==", "=", "!=", "<", "<=", ">", ">=", "in", "not in")
                         (min is None and max is None and not null_count)
                         # Check "is" and "is not" filters first
                         or operator == "is"
```