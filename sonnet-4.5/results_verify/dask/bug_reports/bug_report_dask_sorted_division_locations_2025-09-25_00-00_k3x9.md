# Bug Report: dask.dataframe.io.io.sorted_division_locations - Docstring Examples Fail

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function has docstring examples that demonstrate usage with plain Python lists, but the function crashes with `TypeError: No dispatch for <class 'list'>` when given a list as input.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.io import sorted_division_locations


@given(
    seq=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=200, deadline=5000)
def test_sorted_division_locations_accepts_lists(seq, chunksize):
    """Property: sorted_division_locations should accept plain Python lists."""
    divisions, locations = sorted_division_locations(seq, chunksize=chunksize)
    assert locations[0] == 0
    assert locations[-1] == len(seq)
```

**Failing input**: Any plain Python list, e.g., `['A', 'B', 'C', 'D', 'E', 'F']`

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
divisions, locations = sorted_division_locations(L, chunksize=2)
```

Output:
```
TypeError: No dispatch for <class 'list'>
```

This example is taken directly from the function's docstring, which claims it should produce `(['A', 'C', 'E', 'F'], [0, 2, 4, 6])`.

## Why This Is A Bug

The function's docstring provides five examples demonstrating its usage, all of which use plain Python lists as input. However, the function immediately fails when given a list because the internal `tolist()` function only has dispatchers registered for NumPy arrays and Pandas objects, not for Python's built-in `list` type.

This is a contract violation - the documentation promises functionality that doesn't work. Users following the documented examples will encounter crashes.

## Fix

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -281,7 +281,10 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):

     # Convert from an ndarray to a plain list so that
     # any divisions we extract from seq are plain Python scalars.
-    seq = tolist(seq)
+    if isinstance(seq, list):
+        seq = list(seq)  # Create a copy to avoid mutating the original
+    else:
+        seq = tolist(seq)
     # we use bisect later, so we need sorted.
     seq_unique = sorted(set(seq))
     duplicates = len(seq_unique) < len(seq)
```

This fix checks if the input is already a list and creates a copy if so. Otherwise, it uses the `tolist()` dispatcher for NumPy/Pandas objects. This ensures the function works with all input types shown in the docstring while maintaining backward compatibility with existing code that passes NumPy arrays or Pandas objects.