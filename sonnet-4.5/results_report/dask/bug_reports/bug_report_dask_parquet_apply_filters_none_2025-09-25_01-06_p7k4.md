# Bug Report: dask.dataframe.io.parquet apply_filters TypeError with Partial None Values

**Target**: `dask.dataframe.io.parquet.core.apply_filters`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `apply_filters` function crashes with a TypeError when parquet statistics have only one of min or max as None (partial None values). Despite a comment claiming "min/max cannot be None for remaining checks" (line 531), the code doesn't validate this before performing comparisons, leading to crashes when comparing None with integers.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

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
```

**Failing input**:
```python
parts = [{'id': 0}]
statistics = [{'filter': False, 'columns': [{'name': 'x', 'min': None, 'max': 0, 'null_count': 0}]}]
```

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import apply_filters

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

apply_filters(parts, statistics, [('x', '=', 50)])
```

**Output:**
```
TypeError: '<=' not supported between instances of 'NoneType' and 'int'
```

**Alternative failing case (reversed):**
```python
statistics = [{
    'filter': False,
    'columns': [{
        'name': 'x',
        'min': 0,
        'max': None,
        'null_count': 0
    }]
}]

apply_filters(parts, statistics, [('x', '=', 50)])
```

## Why This Is A Bug

The `apply_filters` function handles parquet row-group statistics for partition-level filtering. Parquet files can legitimately have partial statistics where only min or max is available. The function has a comment on line 531 stating "min/max cannot be None for remaining checks" but fails to enforce this invariant.

The existing None checks (lines 519-529) only handle these cases:
1. Both min and max are None with no null_count
2. Both min and max are None with null_count
3. Special handling for "is" and "is not" operators

They don't handle the case where exactly one of min or max is None, causing the code to fall through to comparison operations (lines 533-547) that crash when comparing None with integers.

This affects real users when:
- Parquet files have incomplete statistics metadata
- Some columns lack min or max statistics
- Legacy parquet files with partial metadata

## Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -528,6 +528,9 @@ def apply_filters(parts, statistics, filters):
                         and max is None
                         and null_count
                         # Start conventional (non-null) filtering
+                        # Skip if either min or max is None for comparison operators
+                        or operator in ("==", "=", "!=", "<", "<=", ">", ">=", "in", "not in")
+                        and (min is None or max is None)
                         # (main/max cannot be None for remaining checks)
                         or operator in ("==", "=")
                         and min <= value <= max
```

Alternatively, a cleaner fix that properly enforces the invariant:

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -516,7 +516,11 @@ def apply_filters(parts, statistics, filters):
                 else:
                     if (
                         # Must allow row-groups with "missing" stats
-                        (min is None and max is None and not null_count)
+                        # If either min or max is None (but not both with null_count),
+                        # we cannot reliably filter, so include the partition
+                        (min is None or max is None)
+                        and not (min is None and max is None and null_count)
+                        # Or both are None with no null_count
                         # Check "is" and "is not" filters first
                         or operator == "is"
                         and null_count
@@ -524,10 +528,6 @@ def apply_filters(parts, statistics, filters):
                         and (not pd.isna(min) or not pd.isna(max))
                         # Allow all-null row-groups if not filtering out nulls
                         or operator != "is not"
-                        and min is None
-                        and max is None
-                        and null_count
-                        # Start conventional (non-null) filtering
                         # (main/max cannot be None for remaining checks)
                         or operator in ("==", "=")
                         and min <= value <= max
```