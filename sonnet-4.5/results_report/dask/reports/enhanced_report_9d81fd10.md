# Bug Report: dask.dataframe.io.parquet.core.apply_filters IndexError on Empty Filters

**Target**: `dask.dataframe.io.parquet.core.apply_filters`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `apply_filters` function crashes with an `IndexError` when passed an empty filters list, failing to handle a valid input case where empty filters should logically mean "no filtering" and return all parts unchanged.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.parquet.core import apply_filters

@given(st.lists(st.dictionaries(
    st.text(min_size=1, max_size=10),
    st.one_of(st.integers(), st.floats(allow_nan=False), st.text())
)))
def test_apply_filters_empty_filters_identity(parts):
    statistics = [{"columns": []} for _ in parts]
    filters = []
    result_parts, result_stats = apply_filters(parts, statistics, filters)
    assert len(result_parts) == len(parts)
    assert len(result_stats) == len(statistics)

if __name__ == "__main__":
    # Run the test
    test_apply_filters_empty_filters_identity()
```

<details>

<summary>
**Failing input**: `parts=[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 17, in <module>
    test_apply_filters_empty_filters_identity()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 5, in test_apply_filters_empty_filters_identity
    st.text(min_size=1, max_size=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 11, in test_apply_filters_empty_filters_identity
    result_parts, result_stats = apply_filters(parts, statistics, filters)
                                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 556, in apply_filters
    conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
                                                      ~~~~~~~^^^
IndexError: list index out of range
Falsifying example: test_apply_filters_empty_filters_identity(
    parts=[],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import apply_filters

# Test case 1: Empty everything (the minimal failing case)
parts = []
statistics = []
filters = []

print("Test case 1: Empty parts, statistics, and filters")
print(f"parts = {parts}")
print(f"statistics = {statistics}")
print(f"filters = {filters}")
print()

try:
    result_parts, result_stats = apply_filters(parts, statistics, filters)
    print("Result:")
    print(f"  result_parts = {result_parts}")
    print(f"  result_stats = {result_stats}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60 + "\n")

# Test case 2: Non-empty parts with empty filters
parts = [{"piece": 1}, {"piece": 2}]
statistics = [{"columns": []}, {"columns": []}]
filters = []

print("Test case 2: Non-empty parts with empty filters")
print(f"parts = {parts}")
print(f"statistics = {statistics}")
print(f"filters = {filters}")
print()

try:
    result_parts, result_stats = apply_filters(parts, statistics, filters)
    print("Result:")
    print(f"  result_parts = {result_parts}")
    print(f"  result_stats = {result_stats}")
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
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/repo.py", line 15, in <module>
    result_parts, result_stats = apply_filters(parts, statistics, filters)
                                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 556, in apply_filters
    conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
                                                      ~~~~~~~^^^
IndexError: list index out of range
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/repo.py", line 38, in <module>
    result_parts, result_stats = apply_filters(parts, statistics, filters)
                                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 556, in apply_filters
    conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
                                                      ~~~~~~~^^^
IndexError: list index out of range
Test case 1: Empty parts, statistics, and filters
parts = []
statistics = []
filters = []

Error: IndexError: list index out of range

============================================================

Test case 2: Non-empty parts with empty filters
parts = [{'piece': 1}, {'piece': 2}]
statistics = [{'columns': []}, {'columns': []}]
filters = []

Error: IndexError: list index out of range
```
</details>

## Why This Is A Bug

The `apply_filters` function crashes when accessing `filters[0]` without first checking if the filters list is empty (line 556 in core.py). This violates expected behavior because:

1. **Empty filters is semantically valid**: An empty filter list naturally represents "no filtering" - the function should return all input parts and statistics unchanged, not crash.

2. **Documentation doesn't forbid empty filters**: The function's docstring states filters should be "List of filters to apply" with type `Union[List[Tuple[str, str, Any]], List[List[Tuple[str, str, Any]]]]`. Empty lists are valid instances of the List type in Python.

3. **Return value specification implies support**: The docstring states the function returns "the same as the input, but possibly a subset". With no filters to apply, returning the complete input is the logical behavior.

4. **Inconsistent with higher-level APIs**: The public `read_parquet` function successfully handles empty filters by converting them to None (line 1701: `filter=_filters_to_expression(filters) if filters else None`), showing that empty filters are expected in normal usage.

5. **Common use case**: Users may build filter lists dynamically based on runtime conditions, resulting in empty lists when no filtering is needed. Requiring special case handling for empty filters adds unnecessary complexity to user code.

## Relevant Context

The crash occurs at line 556 in `/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py`:
```python
conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
```

This line attempts to check the structure of filters (whether it's in DNF form) by accessing `filters[0]`, but doesn't handle the case where `filters` is an empty list.

Interestingly, the Arrow engine's internal `_filters_to_expression` function (line 374-375) explicitly treats empty filters as "Malformed", but this is an implementation detail of a different engine, and the higher-level APIs protect against this by checking `if filters` before calling it.

The codebase shows multiple patterns of defensive programming around filters:
- Line 336: `if filters else set()`
- Line 1290: `if filters is not None:`
- Line 1344: `if filters is not None:`
- Line 1701: `if filters else None`

This demonstrates that empty/None filters are expected edge cases that should be handled gracefully.

## Proposed Fix

```diff
--- a/core.py
+++ b/core.py
@@ -553,6 +553,10 @@ def apply_filters(parts, statistics, filters):

         return parts, statistics

+    # Handle empty filters - return all parts unfiltered
+    if not filters:
+        return parts, statistics
+
     conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]

     out_parts, out_statistics = apply_conjunction(parts, statistics, conjunction)
```