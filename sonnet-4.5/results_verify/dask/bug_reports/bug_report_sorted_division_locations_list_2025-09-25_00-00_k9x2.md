# Bug Report: sorted_division_locations Crashes on Python Lists

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function crashes with a `TypeError` when given Python lists as input, despite the docstring showing multiple examples using Python lists.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.io import sorted_division_locations

@given(
    st.lists(st.integers(), min_size=1),
    st.integers(min_value=1, max_value=100)
)
def test_sorted_division_locations_accepts_lists(seq, chunksize):
    divisions, locations = sorted_division_locations(seq, chunksize=chunksize)
    assert locations[0] == 0
    assert locations[-1] == len(seq)
```

**Failing input**: Any Python list, e.g., `seq=[0], chunksize=1`

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
divisions, locations = sorted_division_locations(L, chunksize=2)
```

Output:
```
Traceback (most recent call last):
  ...
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 774, in dispatch
    raise TypeError(f"No dispatch for {cls}")
TypeError: No dispatch for <class 'list'>
```

## Why This Is A Bug

The function's docstring contains multiple examples that use Python lists as input:

```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])

>>> sorted_division_locations(L, chunksize=3)
(['A', 'D', 'F'], [0, 3, 6])
```

However, when you actually run these examples, they crash with `TypeError: No dispatch for <class 'list'>`.

The issue is in the `tolist` function called at the start of `sorted_division_locations`. The `tolist` dispatch system doesn't have a handler for Python lists, even though converting a list to a list should be trivial (identity function).

## Fix

Add a dispatch handler for Python lists in the `tolist` function:

```diff
--- a/dask/dataframe/dispatch.py
+++ b/dask/dataframe/dispatch.py
@@ -XX,6 +XX,10 @@ def tolist(obj):
     """Convert obj to a list"""
     ...

+@tolist_dispatch.register(list)
+def tolist_list(obj):
+    return obj
+
 @tolist_dispatch.register(np.ndarray)
 def tolist_numpy(obj):
     return obj.tolist()
```

Alternatively, the function could check if the input is already a list before calling `tolist`:

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
```