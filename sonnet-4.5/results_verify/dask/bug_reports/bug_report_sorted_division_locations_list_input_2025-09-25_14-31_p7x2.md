# Bug Report: sorted_division_locations Docstring Examples Use Unsupported List Type

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstring for `sorted_division_locations` contains examples using plain Python lists as input, but the implementation only supports numpy arrays, pandas Series/Index, and similar types. Attempting to use a plain list results in a `TypeError: No dispatch for <class 'list'>`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=1000)
def test_sorted_division_locations_with_list_input(seq, chunksize):
    seq = sorted(seq)
    divisions, locations = sorted_division_locations(seq, chunksize=chunksize)
    assert len(divisions) > 0
```

**Failing input**: `seq=[0]` (or any plain Python list)

## Reproducing the Bug

The docstring shows this example:

```python
>>> L = ['A', 'B', 'C', 'D', 'E', 'F']
>>> sorted_division_locations(L, chunksize=2)
(['A', 'C', 'E', 'F'], [0, 2, 4, 6])
```

However, running this code produces an error:

```python
from dask.dataframe.io.io import sorted_division_locations

L = ['A', 'B', 'C', 'D', 'E', 'F']
result = sorted_division_locations(L, chunksize=2)
```

**Error:**
```
TypeError: No dispatch for <class 'list'>
```

Full traceback shows the error occurs at `tolist(seq)` on line 284 of io.py, which calls a dispatch function that has no handler registered for plain Python lists.

## Why This Is A Bug

This is a **contract violation** - the function's docstring explicitly documents and demonstrates usage with plain Python lists, establishing an API contract that users can rely on. However, the implementation doesn't honor this contract.

Users who follow the documented examples will encounter unexpected failures. This violates the principle that documentation should accurately reflect the code's behavior.

## Fix

There are three possible fixes:

**Option 1** (Preferred): Register a `tolist` dispatch handler for plain lists that simply returns the list unchanged:

```diff
diff --git a/dask/dataframe/backends.py b/dask/dataframe/backends.py
index 1234567..abcdefg 100644
--- a/dask/dataframe/backends.py
+++ b/dask/dataframe/backends.py
@@ -10,6 +10,11 @@ from dask.dataframe.dispatch import (
     tolist_dispatch,
 )

+@tolist_dispatch.register(list)
+def tolist_list(obj):
+    return obj
+
+
 @tolist_dispatch.register((np.ndarray, pd.Series, pd.Index, pd.Categorical))
 def tolist_numpy_or_pandas(obj):
     return obj.tolist()
```

**Option 2**: Update the docstring to remove list examples and clarify that only numpy/pandas types are supported.

**Option 3**: Add explicit list handling in `sorted_division_locations` before calling `tolist`.