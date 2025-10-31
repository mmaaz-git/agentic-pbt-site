# Bug Report: pandas.core.arrays.sparse.SparseArray argmin/argmax Empty Sequence

**Target**: `pandas.core.arrays.sparse.SparseArray.argmin` and `SparseArray.argmax`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`SparseArray.argmin()` and `SparseArray.argmax()` crash with "ValueError: attempt to get argmin/argmax of an empty sequence" when the array contains only fill_value elements (i.e., when `npoints == 0`).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.arrays import SparseArray

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100))
def test_argmin_argmax_no_crash(values):
    arr = np.array(values)
    sparse = SparseArray(arr)

    argmin_result = sparse.argmin()
    argmax_result = sparse.argmax()

    assert argmin_result == arr.argmin()
    assert argmax_result == arr.argmax()
```

**Failing input**: `[0]` (or any array containing only the fill_value)

## Reproducing the Bug

```python
from pandas.arrays import SparseArray
import numpy as np

sparse = SparseArray([0])

result = sparse.argmin()
```

**Output**: `ValueError: attempt to get argmin of an empty sequence`

## Why This Is A Bug

When a SparseArray contains only fill_value elements, `sp_values` is empty (length 0). In the `_argmin_argmax` method (line 1658):

```python
_candidate = non_nan_idx[func(non_nans)]
```

When `non_nans` (which is `sp_values[~mask]`) is empty, calling `np.argmin()` or `np.argmax()` on it raises a ValueError.

The code should handle this case by recognizing that when all elements are fill_value, the argmin/argmax is simply the first index (for ties, numpy returns the first occurrence).

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1654,8 +1654,13 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         idx = np.arange(values.shape[0])
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]
-
-        _candidate = non_nan_idx[func(non_nans)]
+
+        if len(non_nans) == 0:
+            # All values are fill_value, return first index
+            return 0
+
+        _candidate = non_nan_idx[func(non_nans)]
         candidate = index[_candidate]

         if isna(self.fill_value):
```