# Bug Report: sorted_division_locations - TypeError on Plain Python Lists

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function's docstring shows examples using plain Python lists, but the function crashes with `TypeError: No dispatch for <class 'list'>` when called with a list.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=50)
)
def test_sorted_division_locations_basic_properties_chunksize(seq, chunksize):
    seq_sorted = sorted(seq)
    divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)

    assert locations[0] == 0
    assert locations[-1] == len(seq_sorted)
    assert divisions[0] == seq_sorted[0]
    assert divisions[-1] == seq_sorted[-1]
```

**Failing input**: `seq=[0], chunksize=1` (or any list)

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
divisions, locations = sorted_division_locations(L, chunksize=2)
```

Expected output (from docstring):
```python
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

Actual output:
```
TypeError: No dispatch for <class 'list'>
```

## Why This Is A Bug

The function's own docstring (lines 262-277 in `io.py`) provides multiple examples showing it being called with plain Python lists:

```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

However, the function calls `tolist(seq)` on line 284, which uses a dispatcher that only handles `np.ndarray`, `pd.Series`, `pd.Index`, and `pd.Categorical` (see `backends.py:776`), but not plain Python `list` objects.

This violates the API contract established by the docstring examples, making it a documentation/implementation mismatch bug.

## Fix

```diff
--- a/dask/dataframe/backends.py
+++ b/dask/dataframe/backends.py
@@ -776,6 +776,11 @@ def categorical_dtype_pandas(categories=None, ordered=False):
 @tolist_dispatch.register((np.ndarray, pd.Series, pd.Index, pd.Categorical))
 def tolist_numpy_or_pandas(obj):
     return obj.tolist()
+
+
+@tolist_dispatch.register(list)
+def tolist_list(obj):
+    return obj
```