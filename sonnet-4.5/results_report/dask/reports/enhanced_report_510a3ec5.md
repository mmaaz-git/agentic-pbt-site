# Bug Report: dask.dataframe.io.parquet.utils._normalize_index_columns Allows Overlapping Column and Index Names

**Target**: `dask.dataframe.io.parquet.utils._normalize_index_columns`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_normalize_index_columns` function returns overlapping column and index names when both user parameters are None and the data parameters contain overlapping values, violating the function's implicit invariant that column and index names must be disjoint.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.parquet.utils import _normalize_index_columns

@given(
    st.one_of(st.none(), st.text(min_size=1, max_size=10), st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5)),
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
    st.one_of(
        st.none(),
        st.just(False),
        st.text(min_size=1, max_size=10),
        st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5)
    ),
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
)
def test_normalize_index_columns_no_intersection(user_columns, data_columns, user_index, data_index):
    try:
        column_names, index_names = _normalize_index_columns(
            user_columns, data_columns, user_index, data_index
        )
        intersection = set(column_names).intersection(set(index_names))
        assert len(intersection) == 0
    except ValueError as e:
        if "must not intersect" in str(e):
            pass
        else:
            raise

# Run the test
if __name__ == "__main__":
    test_normalize_index_columns_no_intersection()
```

<details>

<summary>
**Failing input**: `user_columns=None, data_columns=['0'], user_index=None, data_index=['0']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 30, in <module>
    test_normalize_index_columns_no_intersection()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 5, in test_normalize_index_columns_no_intersection
    st.one_of(st.none(), st.text(min_size=1, max_size=10), st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5)),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 21, in test_normalize_index_columns_no_intersection
    assert len(intersection) == 0
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_normalize_index_columns_no_intersection(
    user_columns=None,
    data_columns=['0'],
    user_index=None,
    data_index=['0'],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/8/hypo.py:22
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.utils import _normalize_index_columns

# Demonstrate the bug: when both user_columns and user_index are None,
# the function returns overlapping column and index names without validation
user_columns = None
data_columns = ['0']
user_index = None
data_index = ['0']

print("Input parameters:")
print(f"  user_columns={user_columns}")
print(f"  data_columns={data_columns}")
print(f"  user_index={user_index}")
print(f"  data_index={data_index}")
print()

column_names, index_names = _normalize_index_columns(
    user_columns, data_columns, user_index, data_index
)

print("Output:")
print(f"  column_names={column_names}")
print(f"  index_names={index_names}")
print()

intersection = set(column_names).intersection(set(index_names))
print(f"Intersection between columns and indices: {intersection}")
print()

if intersection:
    print("BUG CONFIRMED: Column and index names overlap!")
    print(f"The name(s) {intersection} appear in both columns and indices.")
else:
    print("No bug: Column and index names are disjoint.")

print("\n" + "="*60 + "\n")

# Contrast with user-specified overlap - this correctly raises an error
print("For comparison, when users explicitly specify overlapping names:")
print("  user_columns=['0']")
print("  data_columns=['0']")
print("  user_index=['0']")
print("  data_index=['0']")
print()

try:
    column_names, index_names = _normalize_index_columns(
        ['0'], ['0'], ['0'], ['0']
    )
    print(f"  Output: column_names={column_names}, index_names={index_names}")
except ValueError as e:
    print(f"  Correctly raises ValueError: {e}")
```

<details>

<summary>
BUG CONFIRMED: Column and index names overlap
</summary>
```
Input parameters:
  user_columns=None
  data_columns=['0']
  user_index=None
  data_index=['0']

Output:
  column_names=['0']
  index_names=['0']

Intersection between columns and indices: {'0'}

BUG CONFIRMED: Column and index names overlap!
The name(s) {'0'} appear in both columns and indices.

============================================================

For comparison, when users explicitly specify overlapping names:
  user_columns=['0']
  data_columns=['0']
  user_index=['0']
  data_index=['0']

  Correctly raises ValueError: Specified index and column names must not intersect
```
</details>

## Why This Is A Bug

The function has an implicit invariant that column and index names must not overlap, evidenced by three key behaviors:

1. **Lines 347-348**: When both user_columns and user_index are specified, the function explicitly checks for overlap and raises a ValueError with message "Specified index and column names must not intersect"

2. **Lines 332-336**: When only user_index is specified, the function actively filters data_columns to exclude any names already in the index: `column_names = [x for x in data_columns if x not in index_names]`

3. **Lines 337-341**: When only user_columns is specified, the function actively filters data_index to exclude any names already in the columns: `index_names = [x for x in data_index if x not in column_names]`

However, **lines 349-352** (the else block when both user parameters are None) simply returns the data_columns and data_index without any validation or filtering, allowing overlaps to pass through unchecked. This creates inconsistent behavior where:
- User-specified overlaps are rejected with an error
- Mixed user/data overlaps are filtered to prevent overlap
- Data-only overlaps are silently accepted

This inconsistency causes data ambiguity where the same field name can appear in both the column list and index list, potentially leading to unexpected behavior in downstream operations that assume these sets are disjoint.

## Relevant Context

The function is located in `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/parquet/utils.py` at lines 295-354. While this is an internal/private function (indicated by the leading underscore), it's used by Dask's parquet reading functionality to normalize column and index specifications.

The bug manifests in a specific edge case: when reading parquet files where the metadata specifies the same field name as both a column and an index, AND the user doesn't explicitly specify which fields should be columns or indices. In practice, this situation is likely rare as most parquet files created by pandas or other tools wouldn't have this overlap.

The function's docstring (lines 296-309) doesn't explicitly state the non-overlap requirement, but the implementation pattern strongly suggests this is the intended behavior across all code paths.

## Proposed Fix

```diff
--- a/dask/dataframe/io/parquet/utils.py
+++ b/dask/dataframe/io/parquet/utils.py
@@ -347,8 +347,12 @@ def _normalize_index_columns(user_columns, data_columns, user_index, data_index
         if set(column_names).intersection(index_names):
             raise ValueError("Specified index and column names must not intersect")
     else:
         # Use default columns and index from the metadata
-        column_names = data_columns
         index_names = data_index
+        # Remove any columns that are also in the index to maintain consistency
+        # with the behavior when only one parameter is specified
+        column_names = [x for x in data_columns if x not in index_names]

     return column_names, index_names
```