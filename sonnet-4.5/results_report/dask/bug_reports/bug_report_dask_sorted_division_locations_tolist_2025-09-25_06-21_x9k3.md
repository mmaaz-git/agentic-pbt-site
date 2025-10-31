# Bug Report: dask.dataframe.io sorted_division_locations Docstring Examples Fail

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function's docstring contains examples using plain Python lists, but these examples fail with `TypeError: No dispatch for <class 'list'>` because the `tolist` dispatch function is not registered for Python lists.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=50)
)
def test_sorted_division_locations_with_lists(seq, chunksize):
    seq_sorted = sorted(seq)
    divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)
    assert divisions[0] == seq_sorted[0]
    assert divisions[-1] == seq_sorted[-1]
```

**Failing input**: Any Python list, e.g., `seq=[0], chunksize=1`

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
result = sorted_division_locations(L, chunksize=2)
```

**Output:**
```
TypeError: No dispatch for <class 'list'>
```

This example is taken directly from the function's docstring (line 263 of `dask/dataframe/io/io.py`), which states:
```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

## Why This Is A Bug

The function's implementation calls `tolist(seq)` at line 284 to convert arrays to lists. The `tolist` function in `dask/dataframe/dispatch.py` uses a type dispatcher that is only registered for:
- `np.ndarray`
- `pd.Series`
- `pd.Index`
- `pd.Categorical`
- `cupy.ndarray`

But NOT for plain Python lists. However, the docstring explicitly shows multiple examples using plain Python lists, creating a contract violation where documented usage fails.

## Fix

```diff
--- a/dask/dataframe/backends.py
+++ b/dask/dataframe/backends.py
@@ -773,6 +773,11 @@ def is_float_na_dtype_numpy_or_pandas(obj):
     return np.issubdtype(obj, np.floating)


+@tolist_dispatch.register(list)
+def tolist_list(obj):
+    return obj
+
+
 @tolist_dispatch.register((np.ndarray, pd.Series, pd.Index, pd.Categorical))
 def tolist_numpy_or_pandas(obj):
     return obj.tolist()
```