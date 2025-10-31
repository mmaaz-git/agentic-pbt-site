# Bug Report: dask.dataframe.io.parquet.core.apply_filters Crashes on Empty Filters List

**Target**: `dask.dataframe.io.parquet.core.apply_filters`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `apply_filters` function crashes with an `IndexError` when given an empty filters list, instead of returning the input unchanged as would be expected for "no filtering".

## Property-Based Test

```python
from hypothesis import given, assume, strategies as st, example
import dask.dataframe.io.parquet.core as parquet_core

@given(
    st.lists(st.text(min_size=1), min_size=1, max_size=10),
    st.lists(st.dictionaries(
        st.text(min_size=1),
        st.one_of(st.none(), st.integers(), st.floats(allow_nan=False)),
        min_size=0,
        max_size=10
    ), min_size=1, max_size=10)
)
@example(parts=['part1'], statistics=[{}])  # Force the specific failing input
def test_apply_filters_returns_subset(parts, statistics):
    assume(len(parts) == len(statistics))
    for stats in statistics:
        stats['columns'] = []
    filters = []
    out_parts, out_statistics = parquet_core.apply_filters(parts, statistics, filters)
    assert len(out_parts) <= len(parts)
```

<details>

<summary>
**Failing input**: `parts=['part1'], statistics=[{}], filters=[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 26, in <module>
    test_apply_filters_returns_subset()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 5, in test_apply_filters_returns_subset
    st.lists(st.text(min_size=1), min_size=1, max_size=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 19, in test_apply_filters_returns_subset
    out_parts, out_statistics = parquet_core.apply_filters(parts, statistics, filters)
                                ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 556, in apply_filters
    conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
                                                      ~~~~~~~^^^
IndexError: list index out of range
Falsifying explicit example: test_apply_filters_returns_subset(
    parts=['part1'],
    statistics=[{}],
)
Test failed!
Exception: IndexError: list index out of range

Full traceback:

Failing input: parts=['part1'], statistics=[{}], filters=[]
```
</details>

## Reproducing the Bug

```python
import dask.dataframe.io.parquet.core as parquet_core

parts = ['part1']
statistics = [{'columns': []}]
filters = []

print("Testing apply_filters with empty filters list:")
print(f"parts = {parts}")
print(f"statistics = {statistics}")
print(f"filters = {filters}")
print()

try:
    out_parts, out_statistics = parquet_core.apply_filters(parts, statistics, filters)
    print(f"Result: out_parts = {out_parts}, out_statistics = {out_statistics}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
IndexError: list index out of range at line 556
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/repo.py", line 14, in <module>
    out_parts, out_statistics = parquet_core.apply_filters(parts, statistics, filters)
                                ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py", line 556, in apply_filters
    conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
                                                      ~~~~~~~^^^
IndexError: list index out of range
Testing apply_filters with empty filters list:
parts = ['part1']
statistics = [{'columns': []}]
filters = []

Error occurred: IndexError: list index out of range
```
</details>

## Why This Is A Bug

The function crashes when attempting to access `filters[0]` on line 556 without first checking if the filters list is empty. This violates the expected behavior in multiple ways:

1. **Function contract violation**: The docstring states the function returns "the same as the input, but possibly a subset". An empty filter list logically represents "no filtering criteria", which should return all input unchanged rather than crashing.

2. **Type signature mismatch**: The function accepts `filters: Union[List[Tuple[str, str, Any]], List[List[Tuple[str, str, Any]]]]`. An empty list `[]` is a valid instance of both `List[Tuple]` and `List[List[Tuple]]`, so the function should handle it.

3. **Defensive programming failure**: The function does not validate its inputs before attempting to access list elements. Line 556 attempts `filters[0]` without checking `len(filters) > 0`.

4. **Semantic inconsistency**: Empty filters should semantically mean "apply no filters" (return everything), similar to how an empty WHERE clause in SQL returns all rows. Crashing instead breaks this intuitive expectation.

## Relevant Context

The crash occurs at `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/core.py:556` where the code attempts to determine if filters are in disjunctive normal form (DNF) by checking `isinstance(filters[0], list)`.

In normal Dask usage, this bug is avoided because the caller checks for non-empty filters before calling `apply_filters`:
```python
if self.filters and stats:
    parts, stats = apply_filters(parts, stats, self.filters)
```

However, since the function is:
- Not marked as internal (no underscore prefix)
- Has comprehensive documentation
- Can be imported and called directly

It should handle all valid inputs robustly, including empty lists.

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