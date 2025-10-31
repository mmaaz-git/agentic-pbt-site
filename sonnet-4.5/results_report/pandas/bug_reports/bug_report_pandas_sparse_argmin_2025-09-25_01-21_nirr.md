# Bug Report: pandas.core.sparse.SparseArray argmin/argmax ValueError on All-Fill-Value Arrays

**Target**: `pandas.core.arrays.sparse.SparseArray.argmin` and `pandas.core.arrays.sparse.SparseArray.argmax`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

SparseArray.argmin() and argmax() raise ValueError when called on an array where all values equal the fill_value, instead of returning the expected index like numpy arrays do.

## Property-Based Test

```python
import numpy as np
from pandas.core.arrays.sparse import SparseArray
from hypothesis import given, strategies as st, settings

@settings(max_examples=100)
@given(
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=1, max_value=20)
)
def test_argmin_argmax_all_fill_values(fill_value, size):
    data = [fill_value] * size
    arr = SparseArray(data, fill_value=fill_value)
    dense = np.array(data)

    assert arr.argmin() == dense.argmin()
    assert arr.argmax() == dense.argmax()
```

**Failing input**: `fill_value=0, size=1` (or any array of all fill_values)

## Reproducing the Bug

```python
from pandas.core.arrays.sparse import SparseArray
import numpy as np

arr = SparseArray([0])
print("Sparse argmin:", arr.argmin())

dense = np.array([0])
print("Dense argmin:", dense.argmin())
```

Output:
```
ValueError: attempt to get argmin of an empty sequence
```

Expected output:
```
Sparse argmin: 0
Dense argmin: 0
```

## Why This Is A Bug

When all values in an array are equal, argmin() and argmax() should return 0 (the index of the first element), matching numpy's behavior. This is valid usage - users should be able to call argmin/argmax on any non-empty array.

The bug occurs because the implementation tries to find the argmin/argmax of `sp_values` (the sparse values), which is empty when all values are fill_values. The code doesn't handle this edge case and fails with a ValueError.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1649,6 +1649,11 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
     def _argmin_argmax(self, kind: Literal["argmin", "argmax"]) -> int:
         values = self._sparse_values
         index = self._sparse_index.indices
+
+        # Handle case where all values are fill_value
+        if len(values) == 0:
+            return self._first_fill_value_loc()
+
         mask = np.asarray(isna(values))
         func = np.argmax if kind == "argmax" else np.argmin
```

The fix checks if the sparse values array is empty (which happens when all values are fill_values) and returns the first fill_value location in that case.