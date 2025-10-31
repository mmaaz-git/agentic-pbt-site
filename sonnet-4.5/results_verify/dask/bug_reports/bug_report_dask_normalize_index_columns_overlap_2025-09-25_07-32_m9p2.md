# Bug Report: dask.dataframe.io.parquet.utils._normalize_index_columns Allows Column/Index Overlap

**Target**: `dask.dataframe.io.parquet.utils._normalize_index_columns`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_normalize_index_columns` function returns overlapping column and index names when both user parameters are None and the data parameters contain overlapping values. This violates the function's documented invariant that column and index names must not intersect.

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
```

**Failing input**: `user_columns=None, data_columns=['0'], user_index=None, data_index=['0']`

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.utils import _normalize_index_columns

user_columns = None
data_columns = ['0']
user_index = None
data_index = ['0']

column_names, index_names = _normalize_index_columns(
    user_columns, data_columns, user_index, data_index
)

print(f"column_names: {column_names}")
print(f"index_names: {index_names}")
print(f"Intersection: {set(column_names).intersection(set(index_names))}")
```

Output:
```
column_names: ['0']
index_names: ['0']
Intersection: {'0'}
```

## Why This Is A Bug

When both `user_columns` and `user_index` are None, the function falls through to the else block (lines 349-352) which simply returns `data_columns` and `data_index` without checking for overlap. This violates the documented invariant established by line 347-348, which raises a ValueError when explicit user parameters create an intersection.

The bug creates an inconsistent API: the function rejects user-specified overlapping columns/indices but silently accepts data-provided overlapping columns/indices.

## Fix

```diff
--- a/dask/dataframe/io/parquet/utils.py
+++ b/dask/dataframe/io/parquet/utils.py
@@ -348,8 +348,11 @@ def _normalize_index_columns(user_columns, data_columns, user_index, data_index
             raise ValueError("Specified index and column names must not intersect")
     else:
         # Use default columns and index from the metadata
-        column_names = data_columns
         index_names = data_index
+        # Remove any columns that are also in the index
+        column_names = [x for x in data_columns if x not in index_names]

     return column_names, index_names
```