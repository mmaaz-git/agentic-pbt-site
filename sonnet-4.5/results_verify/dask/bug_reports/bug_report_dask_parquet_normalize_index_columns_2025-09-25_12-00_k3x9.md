# Bug Report: dask.dataframe.io.parquet.utils._normalize_index_columns Allows Overlapping Columns and Index

**Target**: `dask.dataframe.io.parquet.utils._normalize_index_columns`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_normalize_index_columns` function allows columns and index names to overlap when neither `user_columns` nor `user_index` are specified, violating the invariant that columns and indices must not intersect.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.parquet.utils import _normalize_index_columns

@given(
    data_columns=st.lists(st.text(min_size=1), min_size=1, unique=True),
    data_index=st.lists(st.text(min_size=1), unique=True)
)
@settings(max_examples=1000)
def test_normalize_index_columns_no_overlap(data_columns, data_index):
    """Columns and indices in output should never overlap."""
    column_names, index_names = _normalize_index_columns(
        user_columns=None,
        data_columns=data_columns,
        user_index=None,
        data_index=data_index
    )

    overlap = set(column_names).intersection(set(index_names))
    assert not overlap, f"Column and index names must not overlap. Got: {overlap}"
```

**Failing input**:
```python
user_columns=None
data_columns=['a']
user_index=None
data_index=['a']
```

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.utils import _normalize_index_columns

column_names, index_names = _normalize_index_columns(
    user_columns=None,
    data_columns=['a'],
    user_index=None,
    data_index=['a']
)

print(f"column_names={column_names}")
print(f"index_names={index_names}")
print(f"overlap={set(column_names) & set(index_names)}")
```

Output:
```
column_names=['a']
index_names=['a']
overlap={'a'}
```

## Why This Is A Bug

The function explicitly checks for overlap when user specifies both columns and index (line 347-348 in utils.py):

```python
if set(column_names).intersection(index_names):
    raise ValueError("Specified index and column names must not intersect")
```

However, when neither is specified and defaults from metadata are used (line 349-352), no such check is performed:

```python
else:
    # Use default columns and index from the metadata
    column_names = data_columns
    index_names = data_index
```

This is inconsistent behavior. The invariant that columns and indices must not intersect should hold for ALL outputs, not just user-specified ones. A field cannot logically be both a column and an index in a DataFrame.

## Fix

Add the overlap check to the default case as well:

```diff
diff --git a/dask/dataframe/io/parquet/utils.py b/dask/dataframe/io/parquet/utils.py
index 1234567..abcdefg 100644
--- a/dask/dataframe/io/parquet/utils.py
+++ b/dask/dataframe/io/parquet/utils.py
@@ -349,6 +349,8 @@ def _normalize_index_columns(user_columns, data_columns, user_index, data_index
     else:
         # Use default columns and index from the metadata
         column_names = data_columns
         index_names = data_index
+        if set(column_names).intersection(index_names):
+            raise ValueError("Column and index names from metadata must not intersect")

     return column_names, index_names
```