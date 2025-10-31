# Bug Report: sorted_division_locations Rejects Plain Lists

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function raises `TypeError` when given a plain Python list, despite its docstring explicitly showing examples with lists as input.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
    npartitions=st.integers(min_value=1, max_value=20)
)
def test_sorted_division_locations_accepts_lists(seq, npartitions):
    seq_sorted = sorted(seq)
    divisions, locations = sorted_division_locations(seq_sorted, npartitions=npartitions)
    assert len(divisions) == len(locations)
```

**Failing input**: `seq=[0], npartitions=1` (or any list input)

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
result = sorted_division_locations(L, chunksize=2)
```

**Output**:
```
TypeError: No dispatch for <class 'list'>
```

**Expected** (from docstring):
```python
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

## Why This Is A Bug

The function's docstring at lines 257-277 in `dask/dataframe/io/io.py` explicitly shows examples using plain Python lists:

```python
    Examples
    --------

    >>> L = ['A', 'B', 'C', 'D', 'E', 'F']
    >>> sorted_division_locations(L, chunksize=2)
    (['A', 'C', 'E', 'F'], [0, 2, 4, 6])

    >>> sorted_division_locations(L, chunksize=3)
    (['A', 'D', 'F'], [0, 3, 6])

    >>> L = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C']
    >>> sorted_division_locations(L, chunksize=3)
    (['A', 'B', 'C', 'C'], [0, 4, 7, 8])
```

However, the function fails because `tolist(seq)` at line 284 dispatches on the type of `seq` and has no handler for plain Python lists. The dispatch is only registered for `np.ndarray`, `pd.Series`, `pd.Index`, and `pd.Categorical` (see `backends.py:776`).

## Fix

Add a dispatch handler for list and tuple types in `dask/dataframe/backends.py`:

```diff
@tolist_dispatch.register((np.ndarray, pd.Series, pd.Index, pd.Categorical))
def tolist_numpy_or_pandas(obj):
    return obj.tolist()


+@tolist_dispatch.register((list, tuple))
+def tolist_list_or_tuple(obj):
+    return list(obj)
+
+
@is_categorical_dtype_dispatch.register(
    (pd.Series, pd.Index, pd.api.extensions.ExtensionDtype, np.dtype)
)
```

This allows the function to handle lists and tuples by converting tuples to lists and passing lists through unchanged, consistent with the documented API.
