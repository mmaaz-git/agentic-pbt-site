# Bug Report: sorted_division_locations Cannot Handle Python Lists

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function's docstring demonstrates usage with plain Python lists, but the function raises `TypeError: No dispatch for <class 'list'>` when given a Python list as input.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.dask_expr.io import sorted_division_locations


@given(seq=st.lists(st.text(alphabet='ABC', min_size=1, max_size=1), min_size=1, max_size=10))
def test_sorted_division_locations_accepts_python_lists(seq):
    """
    Property: sorted_division_locations should accept plain Python lists.

    The function's docstring provides multiple examples using plain Python lists:
    - L = ['A', 'B', 'C', 'D', 'E', 'F']
    - sorted_division_locations(L, chunksize=2)

    This property verifies the documented behavior works.
    """
    divisions, locations = sorted_division_locations(seq, chunksize=2)

    assert isinstance(divisions, list)
    assert isinstance(locations, list)
    assert locations[0] == 0
    assert locations[-1] == len(seq)
```

**Failing input**: `seq=['A']`

## Reproducing the Bug

```python
from dask.dataframe.dask_expr.io import sorted_division_locations

seq = ['A']
divisions, locations = sorted_division_locations(seq, chunksize=2)
print(f"divisions={divisions}, locations={locations}")
```

**Output**:
```
TypeError: No dispatch for <class 'list'>
```

## Why This Is A Bug

The function's docstring explicitly shows examples using plain Python lists:

```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

However, the implementation calls `tolist(seq)` which uses a dispatch system that only supports `numpy.ndarray`, `pandas.Series`, `pandas.Index`, and `pandas.Categorical` types. The dispatch system raises `TypeError` when given a plain Python `list`.

This is a contract violation - the documented API (via docstring examples) does not match the actual implementation behavior.

## Fix

The simplest fix is to check if the input is already a list before calling `tolist`:

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -281,7 +281,10 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):

     # Convert from an ndarray to a plain list so that
     # any divisions we extract from seq are plain Python scalars.
-    seq = tolist(seq)
+    if isinstance(seq, list):
+        seq = seq
+    else:
+        seq = tolist(seq)
     # we use bisect later, so we need sorted.
     seq_unique = sorted(set(seq))
     duplicates = len(seq_unique) < len(seq)
```

Or more concisely:

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -281,7 +281,7 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):

     # Convert from an ndarray to a plain list so that
     # any divisions we extract from seq are plain Python scalars.
-    seq = tolist(seq)
+    seq = seq if isinstance(seq, list) else tolist(seq)
     # we use bisect later, so we need sorted.
     seq_unique = sorted(set(seq))
     duplicates = len(seq_unique) < len(seq)
```

Alternatively, a handler for `list` type could be registered in `dask/dataframe/dispatch.py`:

```diff
--- a/dask/dataframe/dispatch.py
+++ b/dask/dataframe/dispatch.py
@@ -88,6 +88,11 @@ tolist_dispatch = Dispatch("tolist")


+@tolist_dispatch.register(list)
+def tolist_list(obj):
+    return obj
+
+
 def tolist(obj):
     func = tolist_dispatch.dispatch(type(obj))
     return func(obj)
```