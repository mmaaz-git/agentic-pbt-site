# Bug Report: sorted_division_locations TypeError with Plain Lists

**Target**: `dask.dataframe.dask_expr.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function crashes with a `TypeError` when passed plain Python lists, despite all docstring examples using plain Python lists. The function only works with numpy arrays or pandas Index objects.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import dask.dataframe.dask_expr.io as io_module

@given(st.lists(st.integers(), min_size=1), st.integers(min_value=1, max_value=100))
@settings(max_examples=500)
def test_sorted_division_locations_with_chunksize(seq, chunksize):
    divisions, locations = io_module.sorted_division_locations(sorted(seq), chunksize=chunksize)

    assert isinstance(divisions, list)
    assert isinstance(locations, list)
    assert len(divisions) == len(locations)
    assert locations[0] == 0
    assert locations[-1] == len(seq)
```

**Failing input**: `seq=[0], chunksize=1` (or any plain Python list)

## Reproducing the Bug

```python
import dask.dataframe.dask_expr.io as io_module

L = ['A', 'B', 'C', 'D', 'E', 'F']
result = io_module.sorted_division_locations(L, chunksize=2)
```

**Output**:
```
TypeError: No dispatch for <class 'list'>
```

## Why This Is A Bug

The function's docstring includes multiple examples that use plain Python lists:

```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

However, the implementation calls `tolist(seq)` which only has dispatch registrations for numpy arrays and pandas objects. This violates the documented API contract.

## Fix

The simplest fix is to add a fast-path check for plain lists before calling `tolist`:

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

Alternatively, register a dispatch for plain lists in `dask/dataframe/dispatch.py`:

```diff
--- a/dask/dataframe/dispatch.py
+++ b/dask/dataframe/dispatch.py
@@ -90,6 +90,11 @@ def tolist_numpy_or_pandas(obj):
     return obj.tolist()


+@tolist_dispatch.register(list)
+def tolist_list(obj):
+    return obj
+
+
 tolist_dispatch.register(np.ndarray, tolist_numpy_or_pandas)
 tolist_dispatch.register(pd.Series, tolist_numpy_or_pandas)
 tolist_dispatch.register(pd.Index, tolist_numpy_or_pandas)
```