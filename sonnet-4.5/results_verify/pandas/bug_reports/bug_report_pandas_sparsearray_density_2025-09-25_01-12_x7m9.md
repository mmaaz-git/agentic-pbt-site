# Bug Report: SparseArray.density ZeroDivisionError on Empty Array

**Target**: `pandas.core.arrays.SparseArray.density`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Accessing the `density` property on an empty SparseArray causes a `ZeroDivisionError` instead of returning a sensible value (0.0 or NaN).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.core.arrays as arr
import numpy as np

@given(st.lists(st.integers(min_value=0, max_value=1000), min_size=0, max_size=100))
@settings(max_examples=500)
def test_sparsearray_empty_and_edge_cases(values):
    if len(values) == 0:
        sparse = arr.SparseArray([], fill_value=0)
        assert len(sparse) == 0
        assert sparse.density == 0 or np.isnan(sparse.density)
    else:
        sparse = arr.SparseArray(values, fill_value=0)
        assert len(sparse) == len(values)
```

**Failing input**: `values=[]`

## Reproducing the Bug

```python
import pandas.core.arrays as arr

sparse = arr.SparseArray([], fill_value=0)

density = sparse.density
```

Output:
```
ZeroDivisionError: division by zero
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 708, in density
    return self.sp_index.npoints / self.sp_index.length
```

## Why This Is A Bug

1. Creating an empty SparseArray is valid usage - users should be able to work with empty arrays
2. Accessing a public property should never crash with ZeroDivisionError
3. The density of an empty array has a reasonable semantic meaning (either 0.0 or NaN)
4. This violates the principle of least surprise - users don't expect property access to raise arithmetic errors

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -705,7 +705,9 @@ class SparseArray(OpsMixin, ExtensionArray):
         >>> s.density
         0.6
         """
-        return self.sp_index.npoints / self.sp_index.length
+        if self.sp_index.length == 0:
+            return 0.0
+        return self.sp_index.npoints / self.sp_index.length

     @property
     def npoints(self) -> int:
```

Alternative: Return `np.nan` instead of `0.0` for empty arrays, which might be more semantically correct since the density of an empty set is undefined. However, `0.0` is arguably more practical.