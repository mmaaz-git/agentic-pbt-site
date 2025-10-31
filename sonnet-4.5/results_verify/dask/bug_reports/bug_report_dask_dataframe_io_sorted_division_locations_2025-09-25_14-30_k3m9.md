# Bug Report: dask.dataframe.io.sorted_division_locations Docstring Examples Don't Work

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstring for `sorted_division_locations` shows examples using plain Python lists, but the function only accepts pandas/numpy/GPU array types. All docstring examples fail with `TypeError: No dispatch for <class 'list'>`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.io import sorted_division_locations


@given(
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=50)
)
def test_function_accepts_lists_per_docstring(seq, chunksize):
    seq = sorted(seq)
    divisions, locations = sorted_division_locations(seq, chunksize=chunksize)
    assert divisions[0] == seq[0]
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

## Why This Is A Bug

The function's docstring (lines 256-277 in `dask/dataframe/io/io.py`) contains 5 examples that all use plain Python lists:

```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

However, the implementation (line 284) calls `tolist(seq)`, which only has dispatch registrations for:
- `np.ndarray`
- `pd.Series`
- `pd.Index`
- `pd.Categorical`
- GPU types (cupy, cudf)

Plain Python `list` is not supported. The documentation contradicts the actual behavior, violating the API contract.

## Fix

**Option 1:** Add support for plain lists (simplest fix):

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

**Option 2:** Update docstring to use pandas Series (more work, less user-friendly):

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -259,7 +259,8 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):
     Examples
     --------

-    >>> L = ['A', 'B', 'C', 'D', 'E', 'F']
+    >>> import pandas as pd
+    >>> L = pd.Series(['A', 'B', 'C', 'D', 'E', 'F'])
     >>> sorted_division_locations(L, chunksize=2)
```

**Recommendation:** Option 1 is preferred as it makes the API more user-friendly and matches user expectations from the docstring.