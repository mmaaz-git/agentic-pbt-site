# Bug Report: sorted_division_locations Rejects Python Lists Despite Documentation

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function crashes when given Python lists, despite all docstring examples showing it being called with Python lists. The function internally calls `tolist()` which only accepts pandas Index or numpy array objects, making the documented examples non-functional.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.io import sorted_division_locations


@given(
    seq=st.lists(st.text(min_size=1, max_size=5, alphabet=st.characters(min_codepoint=65, max_codepoint=70)), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=20)
)
def test_sorted_division_locations_docstring_contract(seq, chunksize):
    seq_sorted = sorted(seq)
    divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)
    assert divisions[0] == seq_sorted[0]
    assert divisions[-1] == seq_sorted[-1]
```

**Failing input**: Any Python list, e.g., `seq=['A']`, `chunksize=1`

## Reproducing the Bug

All docstring examples fail:

```python
from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
result = sorted_division_locations(L, chunksize=2)
```

This crashes with:
```
TypeError: No dispatch for <class 'list'>
```

## Why This Is A Bug

The function's docstring contains 5 examples, all using Python lists:

```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

None of these examples work. The issue is on line 284 of `/dask/dataframe/io/io.py`:

```python
seq = tolist(seq)
```

The `tolist()` dispatch function only accepts pandas Index and numpy array types, not Python lists. This violates the documented API contract - users following the documentation will encounter immediate failures.

## Fix

Replace `tolist(seq)` with a simple check that handles lists directly:

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

Or more simply, since `tolist()` is meant to convert to a list, we can just check if it's already a list and skip the conversion:

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -281,7 +281,8 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):

     # Convert from an ndarray to a plain list so that
     # any divisions we extract from seq are plain Python scalars.
-    seq = tolist(seq)
+    if not isinstance(seq, list):
+        seq = tolist(seq)
     # we use bisect later, so we need sorted.
     seq_unique = sorted(set(seq))
     duplicates = len(seq_unique) < len(seq)
```