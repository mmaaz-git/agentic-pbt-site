# Bug Report: sorted_division_locations TypeError with List Input

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function fails with `TypeError: No dispatch for <class 'list'>` when given a Python list as input, despite the function's docstring explicitly showing examples using lists.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=100),
    npartitions=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=500)
def test_sorted_division_locations_invariants(seq, npartitions):
    assume(npartitions <= len(seq))
    divisions, locations = sorted_division_locations(seq, npartitions=npartitions)
    assert len(divisions) == len(locations)
```

**Failing input**: `seq=[0]` (or any list), `npartitions=1`

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
result = sorted_division_locations(L, chunksize=2)
```

**Error:**
```
TypeError: No dispatch for <class 'list'>
    at /home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/io.py:284
```

## Why This Is A Bug

The function's own docstring provides examples using Python lists:

```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

However, when this exact code from the docstring is executed, it raises a `TypeError`. This violates the documented contract of the function.

The issue occurs because the `tolist()` dispatcher at line 284 does not have a registered handler for Python's built-in `list` type. The code appears to be missing a check like:

```python
if isinstance(seq, list):
    pass  # Lists are already lists
else:
    seq = tolist(seq)
```

## Fix

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -279,9 +279,11 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):
     if (npartitions is None) == (chunksize is None):
         raise ValueError("Exactly one of npartitions and chunksize must be specified.")

-    # Convert from an ndarray to a plain list so that
-    # any divisions we extract from seq are plain Python scalars.
-    seq = tolist(seq)
+    # Convert from an ndarray (or other sequence) to a plain list so that
+    # any divisions we extract from seq are plain Python scalars.
+    if not isinstance(seq, list):
+        seq = tolist(seq)
+
     # we use bisect later, so we need sorted.
     seq_unique = sorted(set(seq))
     duplicates = len(seq_unique) < len(seq)
```

This fix adds a check to skip the `tolist()` conversion when the input is already a Python list, making the implementation consistent with the documented examples.