# Bug Report: _sort_mixed Crashes on Arrays with Tuples

**Target**: `dask.dataframe.dask_expr._expr._sort_mixed`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_sort_mixed` function crashes with a `ValueError` when sorting arrays containing tuples because `np.argsort` returns a 2D array for tuples, which cannot be concatenated with the 1D arrays from other types.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from dask.dataframe.dask_expr._expr import _sort_mixed

@given(st.lists(st.one_of(
    st.integers(),
    st.text(),
    st.tuples(st.integers()),
    st.just(None)
), min_size=1, max_size=50))
def test_sort_mixed_order(values):
    arr = np.array(values, dtype=object)
    result = _sort_mixed(arr)
    assert len(result) == len(arr)
```

**Failing input**: `values=[(0,)]`

## Reproducing the Bug

```python
import numpy as np
from dask.dataframe.dask_expr._expr import _sort_mixed

values = np.array([(0,)], dtype=object)
result = _sort_mixed(values)
```

This raises:
```
ValueError: all the input arrays must have same number of dimensions,
but the array at index 0 has 1 dimension(s) and the array at index 1 has 2 dimension(s)
```

## Why This Is A Bug

The function's docstring states it handles mixed types including tuples ("order ints before strings before nulls"), and the implementation explicitly checks for tuples with `tuple_pos`. However, when `np.argsort` is called on an array of tuples, it returns a multi-dimensional array instead of a 1D array of indices, causing `np.concatenate` to fail when combining with other 1D index arrays.

This is a crash on valid input - tuples are explicitly supported by the function but cause it to fail.

## Fix

The issue is that `np.argsort` on object arrays containing tuples produces unexpected results. The fix is to convert tuple indices to a 1D array by flattening or using lexsort:

```diff
--- a/dask/dataframe/dask_expr/_expr.py
+++ b/dask/dataframe/dask_expr/_expr.py
@@ -3896,8 +3896,13 @@ def _sort_mixed(values):
     tuple_pos = np.array([isinstance(x, tuple) for x in values], dtype=bool)
     null_pos = np.array([pd.isna(x) for x in values], dtype=bool)
     num_pos = ~str_pos & ~null_pos & ~tuple_pos
     str_argsort = np.argsort(values[str_pos])
-    tuple_argsort = np.argsort(values[tuple_pos])
+    # np.argsort on tuples returns 2D array, flatten to 1D
+    if tuple_pos.any():
+        tuple_argsort = np.lexsort([values[tuple_pos]])
+    else:
+        tuple_argsort = np.array([], dtype=int)
     num_argsort = np.argsort(values[num_pos])
     # convert boolean arrays to positional indices, then order by underlying values
     str_locs = str_pos.nonzero()[0].take(str_argsort)
```