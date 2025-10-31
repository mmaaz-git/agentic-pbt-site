# Bug Report: dask.dataframe.io.sorted_division_locations TypeError with Plain Lists

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function crashes with `TypeError: No dispatch for <class 'list'>` when called with plain Python lists, despite its docstring containing multiple examples using plain lists.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.io import sorted_division_locations


@given(
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=1000)
def test_sorted_division_locations_accepts_lists(seq, chunksize):
    seq_sorted = sorted(seq)
    divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)
    assert divisions[0] == seq_sorted[0]
    assert divisions[-1] == seq_sorted[-1]
```

**Failing input**: `seq=[0], chunksize=1` (or any plain Python list)

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
divisions, locations = sorted_division_locations(L, chunksize=2)
```

Running this code produces:
```
TypeError: No dispatch for <class 'list'>
```

This exact example is taken from the function's docstring (lines 262-264 of io.py), which claims it should return `(['A', 'C', 'E', 'F'], [0, 2, 4, 6])`.

## Why This Is A Bug

The function's docstring explicitly shows examples using plain Python lists as input. However, the implementation calls `tolist(seq)` on line 284, which uses a dispatch mechanism that only supports `(np.ndarray, pd.Series, pd.Index, pd.Categorical)` types, not plain Python lists. This causes the function to crash on inputs shown in its own documentation.

## Fix

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -281,7 +281,9 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):

     # Convert from an ndarray to a plain list so that
     # any divisions we extract from seq are plain Python scalars.
-    seq = tolist(seq)
+    if not isinstance(seq, list):
+        seq = tolist(seq)
+
     # we use bisect later, so we need sorted.
     seq_unique = sorted(set(seq))
     duplicates = len(seq_unique) < len(seq)
```