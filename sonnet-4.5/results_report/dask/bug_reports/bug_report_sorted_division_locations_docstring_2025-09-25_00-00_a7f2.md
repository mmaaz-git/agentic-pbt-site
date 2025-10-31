# Bug Report: sorted_division_locations Docstring Examples Fail

**Target**: `dask.dataframe.dask_expr.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstring examples for `sorted_division_locations` use plain Python lists, but the function raises `TypeError: No dispatch for <class 'list'>` when given a list. The function only works with numpy arrays, pandas Series, or pandas Index objects, contradicting its documentation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.dask_expr.io import sorted_division_locations

@given(
    st.lists(st.text(alphabet="ABCDEF", min_size=1, max_size=1), min_size=1),
    st.integers(min_value=1, max_value=10)
)
def test_sorted_division_locations_accepts_lists(seq, chunksize):
    sorted_seq = sorted(seq)
    divisions, locations = sorted_division_locations(sorted_seq, chunksize=chunksize)
    assert isinstance(divisions, list)
    assert isinstance(locations, list)
```

**Failing input**: `seq=['A', 'B', 'C'], chunksize=2`

## Reproducing the Bug

This is the first example from the function's own docstring:

```python
from dask.dataframe.dask_expr.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
result = sorted_division_locations(L, chunksize=2)
```

**Output:**
```
TypeError: No dispatch for <class 'list'>
```

The function works with numpy arrays or pandas objects:

```python
import numpy as np
from dask.dataframe.dask_expr.io import sorted_division_locations

L = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
result = sorted_division_locations(L, chunksize=2)
print(result)
```

**Output:**
```
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

## Why This Is A Bug

The function's docstring contains five examples that all use plain Python lists:

```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

None of these examples work with the current implementation. This violates the API contract established by the documentation. Users reading the docstring would reasonably expect the function to accept lists, but it fails immediately.

## Fix

The issue is in line 284 of `sorted_division_locations` where it calls `tolist(seq)`, which is a dispatch function that doesn't handle plain Python lists. The simplest fix is to check if the input is already a list and skip the conversion:

```diff
def sorted_division_locations(seq, npartitions=None, chunksize=None):
    """Find division locations and values in sorted list
    ...
    """
    if (npartitions is None) == (chunksize is None):
        raise ValueError("Exactly one of npartitions and chunksize must be specified.")

-   # Convert from an ndarray to a plain list so that
-   # any divisions we extract from seq are plain Python scalars.
-   seq = tolist(seq)
+   # Convert from an ndarray to a plain list so that
+   # any divisions we extract from seq are plain Python scalars.
+   if not isinstance(seq, list):
+       seq = tolist(seq)
    # we use bisect later, so we need sorted.
    seq_unique = sorted(set(seq))
    ...
```

This preserves the behavior for numpy/pandas inputs while allowing plain lists (as documented).