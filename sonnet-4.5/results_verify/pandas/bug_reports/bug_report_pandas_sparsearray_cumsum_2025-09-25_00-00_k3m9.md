# Bug Report: pandas.core.sparse.SparseArray.cumsum() Infinite Recursion

**Target**: `pandas.core.sparse.api.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Calling `cumsum()` on a SparseArray causes infinite recursion and crashes with RecursionError when the fill value is not null.

## Property-Based Test

```python
import numpy as np
from pandas.core.sparse.api import SparseArray
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100))
def test_cumsum_preserves_length(data):
    sparse = SparseArray(data)
    result = sparse.cumsum()
    assert len(result) == len(sparse), \
        f"cumsum() changed length: {len(result)} != {len(sparse)}"
```

**Failing input**: `[0]` (or any list with non-null fill value)

## Reproducing the Bug

```python
from pandas.core.sparse.api import SparseArray

sparse = SparseArray([1, 2, 3])
result = sparse.cumsum()
```

Output:
```
RecursionError: maximum recursion depth exceeded
```

## Why This Is A Bug

The `cumsum()` method is a documented public API method that should compute the cumulative sum. Instead, it crashes with infinite recursion for arrays with non-null fill values. The bug occurs because line 1550 in `array.py`:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()
```

This creates a new SparseArray from the dense array, then calls `cumsum()` on it. However, the new SparseArray also has `_null_fill_value = False`, causing it to recursively call the same line again, leading to infinite recursion.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         if axis is not None and axis >= self.ndim:  # Mimic ndarray behaviour.
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return SparseArray(self.to_dense().cumsum())

         return SparseArray(
             self.sp_values.cumsum(),
```

The fix is to call `cumsum()` on the dense numpy array first, then wrap the result in a SparseArray, rather than wrapping in a SparseArray first and then calling `cumsum()` on it.