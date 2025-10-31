# Bug Report: pandas.core.arrays.sparse.SparseArray.cumsum Infinite Recursion

**Target**: `pandas.core.arrays.sparse.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`SparseArray.cumsum()` enters infinite recursion when called on arrays with non-NA fill values (e.g., fill_value=0 for integers), causing a RecursionError and making the method completely unusable for such arrays.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st
from pandas.arrays import SparseArray


@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50))
def test_cumsum_matches_dense(data):
    arr = SparseArray(data)
    dense = arr.to_dense()

    sparse_cumsum = arr.cumsum()
    dense_cumsum = np.cumsum(dense)

    assert np.array_equal(sparse_cumsum.to_dense(), dense_cumsum)
```

**Failing input**: `data=[0, 1, 2]` (or any list with default fill_value=0)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.arrays import SparseArray

arr = SparseArray([0, 1, 2], fill_value=0)
arr.cumsum()
```

Output:
```
RecursionError: maximum recursion depth exceeded
```

## Why This Is A Bug

At line 1549-1550, when `not self._null_fill_value`, the code calls:
```python
return SparseArray(self.to_dense()).cumsum()
```

This creates a new SparseArray from the dense representation without specifying a fill_value. The new SparseArray defaults to fill_value=0 for integer dtypes, which means `_null_fill_value` is still False. This triggers the same code path again, causing infinite recursion.

The method is completely broken for any array with non-NA fill values, which is the common case for integer and boolean sparse arrays.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,8 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            # Convert to dense, compute cumsum, then convert back with NaN fill to avoid recursion
+            return type(self)(np.cumsum(self.to_dense()), fill_value=np.nan)

         return SparseArray(
             self.sp_values.cumsum(),
```