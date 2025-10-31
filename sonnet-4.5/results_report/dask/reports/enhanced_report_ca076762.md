# Bug Report: dask.dataframe.io.parquet.core.apply_filters IndexError on Empty Filters List

**Target**: `dask.dataframe.io.parquet.core.apply_filters`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `apply_filters` function crashes with an `IndexError` when called with an empty `filters` list, despite this being a valid input according to the function's documented behavior.

## Property-Based Test

```python
from hypothesis import given, assume, strategies as st
import dask.dataframe.io.parquet.core as core

@given(
    st.lists(st.text(), min_size=0, max_size=20),
    st.lists(st.dictionaries(st.text(), st.integers()), min_size=0, max_size=20)
)
def test_apply_filters_empty_filters_returns_all(parts, statistics):
    assume(len(parts) == len(statistics))

    filters = []

    filtered_parts, filtered_stats = core.apply_filters(parts, statistics, filters)

    assert filtered_parts == parts
    assert filtered_stats == statistics

# Run the test
if __name__ == "__main__":
    test_apply_filters_empty_filters_returns_all()
```

<details>

<summary>
**Failing input**: `parts=[], statistics=[]`
</summary>
```
Falsifying example: test_apply_filters_empty_filters_returns_all(
    parts=[],
    statistics=[],
)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 20, in <module>
    test_apply_filters_empty_filters_returns_all()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 5, in test_apply_filters_empty_filters_returns_all
    st.lists(st.text(), min_size=0, max_size=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 13, in test_apply_filters_empty_filters_returns_all
    filtered_parts, filtered_stats = core.apply_filters(parts, statistics, filters)
                                     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 556, in apply_filters
    conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
                                                      ~~~~~~~^^^
IndexError: list index out of range
```
</details>

## Reproducing the Bug

```python
import dask.dataframe.io.parquet.core as core

# Test with empty filters list - the reported issue
parts = []
statistics = []
filters = []

print("Testing apply_filters with empty filters list...")
print(f"Input: parts={parts}, statistics={statistics}, filters={filters}")

try:
    filtered_parts, filtered_stats = core.apply_filters(parts, statistics, filters)
    print(f"Success: filtered_parts={filtered_parts}, filtered_stats={filtered_stats}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
IndexError: list index out of range
</summary>
```
Testing apply_filters with empty filters list...
Input: parts=[], statistics=[], filters=[]
Error: IndexError: list index out of range
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/repo.py", line 12, in <module>
    filtered_parts, filtered_stats = core.apply_filters(parts, statistics, filters)
                                     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 556, in apply_filters
    conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
                                                      ~~~~~~~^^^
IndexError: list index out of range
```
</details>

## Why This Is A Bug

The function's docstring explicitly states that it returns "parts, statistics: the same as the input, but possibly a subset". The phrase "possibly a subset" clearly indicates that the function may return the full input when no filtering conditions apply. An empty filter list logically represents "no filtering conditions", and therefore should return all input parts and statistics unchanged.

The crash occurs at line 556 in the function where it attempts to access `filters[0]` without first checking if the filters list is empty. This violates the documented contract and reasonable user expectations. Users commonly build filters conditionally in their code, and an empty filter list is a natural outcome when no filtering conditions are met. For example:

```python
filters = []
if user_wants_recent:
    filters.append(("date", ">", "2025-01-01"))
if user_wants_active:
    filters.append(("status", "==", "active"))
# filters could be empty if no conditions were selected
result = apply_filters(parts, stats, filters)  # Should work but crashes
```

The function already contains logic to handle filters that don't match any data (returning all input), so an empty filter list should be treated equivalently.

## Relevant Context

The `apply_filters` function is located in `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py`. It's designed to perform partition-level filtering on parquet files to prevent unnecessary loading of row-groups and files.

The function supports two filter formats:
1. Simple list of tuples: `[('x', '>', 5), ('y', '==', 10)]` (AND conjunction)
2. Disjunctive Normal Form (DNF): `[[('x', '>', 5)], [('y', '==', 10)]]` (OR of ANDs)

The bug affects both empty parts/statistics and non-empty ones - it crashes regardless of the data as long as the filters list is empty.

Documentation reference: The function is part of Dask's parquet reading functionality, used internally when `read_parquet` is called with filter parameters.

## Proposed Fix

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -553,6 +553,9 @@ def apply_filters(parts, statistics, filters):

         return parts, statistics

+    if not filters:
+        return parts, statistics
+
     conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]

     out_parts, out_statistics = apply_conjunction(parts, statistics, conjunction)
```