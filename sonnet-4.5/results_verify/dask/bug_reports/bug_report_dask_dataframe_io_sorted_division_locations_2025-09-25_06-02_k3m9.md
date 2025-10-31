# Bug Report: sorted_division_locations Doesn't Support Plain Python Lists

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function's docstring contains examples showing it being used with plain Python lists, but the actual implementation crashes with `TypeError: No dispatch for <class 'list'>` when passed a Python list. This is a contract violation where the documented API differs from the implementation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=1000)
def test_sorted_division_locations_basic_invariants(seq, chunksize):
    seq_sorted = sorted(seq)
    divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)

    assert len(divisions) == len(locations)
    assert locations[0] == 0
    assert locations[-1] == len(seq_sorted)
```

**Failing input**: `seq=[0], chunksize=1` (or any Python list)

## Reproducing the Bug

The docstring shows this example:

```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

But running this code produces:

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
result = sorted_division_locations(L, chunksize=2)
```

**Error:**
```
TypeError: No dispatch for <class 'list'>
  File "dask/dataframe/io/io.py", line 284, in sorted_division_locations
    seq = tolist(seq)
  File "dask/dataframe/dispatch.py", line 91, in tolist
    func = tolist_dispatch.dispatch(type(obj))
  File "dask/utils.py", line 774, in dispatch
    raise TypeError(f"No dispatch for {cls}")
```

## Why This Is A Bug

The function calls `tolist(seq)` on line 284 of `dask/dataframe/io/io.py`. The `tolist` dispatch function is only registered for:
- `np.ndarray`
- `pd.Series`
- `pd.Index`
- `pd.Categorical`
- `cudf` arrays
- `cupy` arrays

But it is NOT registered for plain Python lists, even though the docstring examples clearly show it should work with lists.

This violates the contract established by the documentation - users following the documented examples will encounter a crash.

## Fix

Register a tolist implementation for Python lists in `dask/dataframe/backends.py`:

```diff
--- a/dask/dataframe/backends.py
+++ b/dask/dataframe/backends.py
@@ -XXX,6 +XXX,11 @@
 def tolist_numpy_or_pandas(obj):
     return obj.tolist()

+
+@tolist_dispatch.register(list)
+def tolist_list(obj):
+    return obj
+
```

Alternatively, if the function was never intended to work with plain Python lists, the docstring examples should be updated to use numpy arrays or pandas Series instead of Python lists.