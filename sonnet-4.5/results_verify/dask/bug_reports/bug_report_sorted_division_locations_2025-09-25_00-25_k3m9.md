# Bug Report: sorted_division_locations fails with plain Python lists despite docstring examples

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function's docstring contains examples using plain Python lists, but the function raises a `TypeError` when called with lists because the internal `tolist` dispatch doesn't support the `list` type.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(), min_size=1),
    chunksize=st.integers(min_value=1, max_value=1000)
)
@settings(max_examples=500)
def test_boundary_invariants(seq, chunksize):
    seq_sorted = sorted(seq)
    divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)

    assert divisions[0] == seq_sorted[0], "First division should be first element"
    assert divisions[-1] == seq_sorted[-1], "Last division should be last element"
    assert locations[0] == 0, "First location should be 0"
    assert locations[-1] == len(seq_sorted), "Last location should be length of seq"
```

**Failing input**: `seq=[0], chunksize=1` (or any list)

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
divisions, locations = sorted_division_locations(L, chunksize=2)
```

**Error:**
```
TypeError: No dispatch for <class 'list'>
```

This is the exact example from the function's own docstring:
```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

## Why This Is A Bug

The function's docstring explicitly documents that it accepts Python lists and provides multiple examples using lists. However, when you run those exact examples, they fail with a `TypeError`. This is a contract violation - the documented API doesn't match the actual behavior.

The root cause is that `sorted_division_locations` calls `tolist(seq)` on line 284 of `dask/dataframe/io/io.py`, and the `tolist_dispatch` function only has registered handlers for:
- `numpy.ndarray`
- `pandas.core.series.Series`
- `pandas.core.indexes.base.Index`
- `pandas.core.arrays.categorical.Categorical`

But NOT for plain Python `list` or `tuple` types.

## Fix

Add a dispatch registration for Python lists in `dask/dataframe/dispatch.py`:

```diff
@tolist_dispatch.register((np.ndarray, pd.Series, pd.Index, pd.Categorical))
def tolist_numpy_or_pandas(obj):
    return obj.tolist()

+@tolist_dispatch.register(list)
+def tolist_list(obj):
+    return obj
```

Alternatively, the fix could be applied directly in `sorted_division_locations` to handle lists before calling `tolist`:

```diff
def sorted_division_locations(seq, npartitions=None, chunksize=None):
    if (npartitions is None) == (chunksize is None):
        raise ValueError("Exactly one of npartitions and chunksize must be specified.")

-    seq = tolist(seq)
+    if isinstance(seq, list):
+        pass
+    else:
+        seq = tolist(seq)
```

The first solution (registering list type) is preferred as it's more general and will fix the issue for all callers of `tolist`.