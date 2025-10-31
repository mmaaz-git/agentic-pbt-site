# Bug Report: sorted_division_locations Rejects Lists Despite Documentation

**Target**: `dask.dataframe.dask_expr.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function's docstring contains examples using plain Python lists as input, but the function raises a TypeError when called with lists. It only accepts numpy arrays or pandas Series.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import dask.dataframe.dask_expr.io as io_module

@given(st.lists(st.integers(), min_size=1))
def test_sorted_division_locations_with_chunksize(seq):
    assume(len(seq) >= 1)
    seq_sorted = sorted(seq)
    chunksize = max(1, len(seq) // 3) if len(seq) > 1 else 1

    divisions, locations = io_module.sorted_division_locations(seq_sorted, chunksize=chunksize)

    assert locations[0] == 0
    assert locations[-1] == len(seq_sorted)
```

**Failing input**: `[0]` (or any list)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
sorted_division_locations(L, chunksize=2)
```

Output:
```
TypeError: No dispatch for <class 'list'>
```

This contradicts the docstring example:
```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

## Why This Is A Bug

The function's own docstring shows it being called with plain Python lists, but it rejects them with a TypeError. This violates the API contract established by the documentation. Users following the documented examples will encounter errors.

The function works correctly with numpy arrays and pandas Series:
```python
import numpy as np
import pandas as pd

sorted_division_locations(np.array(['A', 'B', 'C', 'D', 'E', 'F']), chunksize=2)
sorted_division_locations(pd.Series(['A', 'B', 'C', 'D', 'E', 'F']), chunksize=2)
```

## Fix

The fix requires adding support for plain Python lists to the `tolist` dispatch function, or updating the docstring to reflect that only numpy arrays and pandas Series are accepted. The former is preferred to maintain backwards compatibility with the documented API.

One approach is to check for list type explicitly in the `sorted_division_locations` function before calling `tolist`:

```diff
--- a/dask/dataframe/io/io.py
+++ b/dask/dataframe/io/io.py
@@ -281,6 +281,10 @@ def sorted_division_locations(seq, npartitions=None, chunksize=None):
     if (npartitions is None) == (chunksize is None):
         raise ValueError("Exactly one of npartitions and chunksize must be specified.")

+    # Support plain Python lists as shown in docstring
+    if isinstance(seq, list):
+        seq = seq  # Already a list, no conversion needed
+    else:
+        seq = tolist(seq)
-    # Convert from an ndarray to a plain list so that
-    # any divisions we extract from seq are plain Python scalars.
-    seq = tolist(seq)
```

Alternatively, register a list handler in the tolist dispatch function.