# Bug Report: sorted_division_locations Docstring Examples Fail

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function's docstring contains multiple examples using plain Python lists, but the function fails with `TypeError: No dispatch for <class 'list'>` when given a list. The implementation expects pandas/cudf Series, contradicting the documented API.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1).map(sorted),
    chunksize=st.integers(min_value=1, max_value=100)
)
def test_accepts_lists_as_documented(seq, chunksize):
    divisions, locations = sorted_division_locations(seq, chunksize=chunksize)
    assert len(divisions) == len(locations)
```

**Failing input**: `seq=[0], chunksize=1` (or any list)

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
divisions, locations = sorted_division_locations(L, chunksize=2)
```

**Output:**
```
TypeError: No dispatch for <class 'list'>
```

This example is taken directly from the function's docstring (line 262-264 in io.py) and should return `(['A', 'C', 'E', 'F'], [0, 2, 4, 6])`.

## Why This Is A Bug

The function's docstring contains 5 examples, all using plain Python lists:
- `sorted_division_locations(['A', 'B', 'C', 'D', 'E', 'F'], chunksize=2)`
- `sorted_division_locations(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C'], chunksize=3)`
- `sorted_division_locations(['A'], chunksize=2)`

However, the implementation calls `tolist(seq)` at line 284, which only supports pandas/cudf Series and numpy arrays - not plain Python lists. This violates the documented API contract.

## Fix

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -281,7 +281,10 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):

     # Convert from an ndarray to a plain list so that
     # any divisions we extract from seq are plain Python scalars.
-    seq = tolist(seq)
+    if isinstance(seq, list):
+        seq = list(seq)  # Make a copy for safety
+    else:
+        seq = tolist(seq)
     # we use bisect later, so we need sorted.
     seq_unique = sorted(set(seq))
     duplicates = len(seq_unique) < len(seq)
```