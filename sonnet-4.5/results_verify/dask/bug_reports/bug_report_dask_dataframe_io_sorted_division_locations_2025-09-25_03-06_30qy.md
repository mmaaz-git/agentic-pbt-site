# Bug Report: dask.dataframe.io.sorted_division_locations Input Type Mismatch

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstring for `sorted_division_locations` shows examples using plain Python lists, but the implementation only accepts pandas Series, numpy arrays, or other types registered with the `tolist_dispatch`. This creates a mismatch between the documented API and actual behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(seq=st.lists(st.text(min_size=1, max_size=1), min_size=1, max_size=10))
def test_sorted_division_locations_accepts_lists_as_documented(seq):
    from dask.dataframe.io.io import sorted_division_locations

    seq_sorted = sorted(seq)
    divisions, locations = sorted_division_locations(seq_sorted, chunksize=2)

    assert divisions[0] == seq_sorted[0]
    assert divisions[-1] == seq_sorted[-1]
```

**Failing input**: `['A', 'B', 'C']` (or any Python list)

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
result = sorted_division_locations(L, chunksize=2)
```

Output:
```
TypeError: No dispatch for <class 'list'>
```

The docstring explicitly shows this exact code as an example:
```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

However, the implementation calls `tolist(seq)` at line 284, which requires `seq` to be a type registered with `tolist_dispatch` (pandas Series, numpy array, etc.).

## Why This Is A Bug

The function's docstring provides executable examples that are part of the API contract. Users reading the documentation will naturally try to use the function with Python lists as shown in the examples, but the code will fail with a TypeError. This violates the principle of least surprise and makes the documentation misleading.

## Fix

Add support for plain Python lists in the `tolist` dispatch, or update the docstring to show that pandas Series or numpy arrays are required:

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -281,6 +281,10 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):

     # Convert from an ndarray to a plain list so that
     # any divisions we extract from seq are plain Python scalars.
+    # Handle plain Python lists
+    if isinstance(seq, list):
+        seq = seq
+    else:
-    seq = tolist(seq)
+        seq = tolist(seq)
     # we use bisect later, so we need sorted.
     seq_unique = sorted(set(seq))
```

Or better yet, register a handler in `dispatch.py`:

```diff
--- a/dask/dataframe/dispatch.py
+++ b/dask/dataframe/dispatch.py
@@ -88,6 +88,11 @@ def categorical_dtype(meta, categories=None, ordered=False):
     return func(categories=categories, ordered=ordered)


+@tolist_dispatch.register(list)
+def tolist_python_list(obj):
+    return obj
+
+
 def tolist(obj):
     func = tolist_dispatch.dispatch(type(obj))
     return func(obj)
```