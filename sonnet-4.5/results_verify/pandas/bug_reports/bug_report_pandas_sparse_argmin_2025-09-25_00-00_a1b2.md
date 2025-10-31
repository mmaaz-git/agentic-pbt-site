# Bug Report: pandas.core.arrays.sparse.SparseArray.argmin Crash on All-Fill-Value Arrays

**Target**: `pandas.core.arrays.sparse.SparseArray.argmin` and `argmax`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`SparseArray.argmin()` and `argmax()` crash with ValueError when called on arrays where all values equal the non-NA fill value, because they attempt to compute argmin/argmax on an empty array of sparse values.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st
from pandas.arrays import SparseArray


@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50))
def test_argmin_argmax_match_dense(data):
    arr = SparseArray(data, fill_value=0)
    dense = arr.to_dense()

    assert arr.argmin() == dense.argmin()
    assert arr.argmax() == dense.argmax()
```

**Failing input**: `data=[0]`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.arrays import SparseArray

arr = SparseArray([0], fill_value=0)
arr.argmin()
```

Output:
```
ValueError: attempt to get argmin of an empty sequence
```

Expected: Should return 0 (matching the behavior of numpy arrays).

## Why This Is A Bug

When all elements in a SparseArray equal the fill_value, the sparse representation stores no values (`sp_values` is empty). The `_argmin_argmax` method at line 1658 calls `np.argmin(non_nans)` without checking if `non_nans` is empty, causing a crash.

This violates the expected behavior that `argmin()` and `argmax()` should work on any non-empty array, matching NumPy's behavior.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1654,8 +1654,16 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         idx = np.arange(values.shape[0])
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]
-
-        _candidate = non_nan_idx[func(non_nans)]
+
+        if len(non_nans) == 0:
+            # All sparse values are NaN, or there are no sparse values at all
+            _loc = self._first_fill_value_loc()
+            if _loc == -1:
+                # Empty array
+                raise ValueError(f"attempt to get {kind} of an empty sequence")
+            return _loc
+
+        _candidate = non_nan_idx[func(non_nans)]
         candidate = index[_candidate]

         if isna(self.fill_value):
```