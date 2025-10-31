# Bug Report: pandas.core.arrays.sparse.SparseArray.cumsum Infinite Recursion

**Target**: `pandas.core.arrays.sparse.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`SparseArray.cumsum()` causes infinite recursion (RecursionError) when the fill_value is not NA/NaN, which is the default behavior for integer arrays (fill_value=0).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.arrays import SparseArray

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100))
def test_cumsum_does_not_crash(values):
    arr = np.array(values)
    sparse = SparseArray(arr)

    cumsum_result = sparse.cumsum()
    assert len(cumsum_result) == len(sparse)
```

**Failing input**: `[1]` (or any list of integers)

## Reproducing the Bug

```python
from pandas.arrays import SparseArray
import numpy as np

sparse = SparseArray([1, 2, 3])
result = sparse.cumsum()
```

**Output**: `RecursionError: maximum recursion depth exceeded`

## Why This Is A Bug

Looking at the implementation (lines 1549-1550 in `array.py`):

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()
```

When `_null_fill_value` is False (i.e., fill_value is not NA), the code:
1. Converts the sparse array to dense
2. Creates a new SparseArray from the dense array
3. Calls cumsum() on the new SparseArray

However, the newly created SparseArray **also has a non-null fill_value** (it inherits the same fill_value=0 for integer arrays). This causes it to enter the same code path, creating another SparseArray and calling cumsum() again, leading to infinite recursion.

This affects all integer SparseArrays (default fill_value=0) and any SparseArray with a non-NA fill_value.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return SparseArray(self.to_dense().cumsum())

         return SparseArray(
             self.sp_values.cumsum(),
```