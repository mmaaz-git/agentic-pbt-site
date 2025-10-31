# Bug Report: dask.dataframe.io.sorted_division_locations - TypeError with Python lists

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: High
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function's docstring contains examples using Python lists, but the implementation raises `TypeError: No dispatch for <class 'list'>` when called with Python lists. The function only accepts pandas or numpy types, contradicting its own documentation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1),
    chunksize=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=500)
def test_divisions_locations_same_length(seq, chunksize):
    seq = sorted(seq)
    divisions, locations = sorted_division_locations(seq, chunksize=chunksize)
    assert len(divisions) == len(locations)
```

**Failing input**: `seq=[0], chunksize=1` (any list fails)

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

This is the exact example from the function's own docstring (line 263-264 in io.py):
```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

## Why This Is A Bug

This violates the API contract in multiple ways:

1. **Docstring examples don't work**: All five examples in the function's docstring use Python lists, but the function raises TypeError when called with lists.

2. **No type hints or documentation about input requirements**: The function signature is `sorted_division_locations(seq, npartitions=None, chunksize=None)` with no indication that `seq` must be a pandas/numpy type.

3. **Implementation changed but documentation didn't**: Line 284 calls `seq = tolist(seq)` which uses a dispatch system that doesn't support Python lists, but the docstring was never updated.

The function is clearly intended to work with Python lists based on its documentation, but the implementation contradicts this.

## Fix

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -281,7 +281,10 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):

     # Convert from an ndarray to a plain list so that
     # any divisions we extract from seq are plain Python scalars.
-    seq = tolist(seq)
+    if isinstance(seq, list):
+        pass  # Already a list
+    else:
+        seq = tolist(seq)
     # we use bisect later, so we need sorted.
     seq_unique = sorted(set(seq))
     duplicates = len(seq_unique) < len(seq)
```