# Bug Report: SparseArray.astype() Returns ndarray Instead of SparseArray

**Target**: `pandas.core.arrays.sparse.array.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Calling `astype()` with a non-SparseDtype returns a numpy ndarray instead of a SparseArray, violating the documented contract that states "The output will always be a SparseArray."

## Property-Based Test

```python
import numpy as np
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100))
def test_astype_returns_sparse_array(data):
    arr = SparseArray(data, fill_value=0)
    result = arr.astype(np.float64)

    assert isinstance(result, SparseArray), \
        f"Expected SparseArray, got {type(result).__name__}"
```

**Failing input**: `SparseArray([0, 1, 2], fill_value=0)` with `dtype=np.float64`

## Reproducing the Bug

```python
import numpy as np
from pandas.arrays import SparseArray

arr = SparseArray([0, 1, 2], fill_value=0)
result = arr.astype(np.float64)

print(type(result))
```

Output:
```
<class 'numpy.ndarray'>
```

Expected:
```
<class 'pandas.core.arrays.sparse.array.SparseArray'>
```

## Why This Is A Bug

The documentation at line 1241 explicitly states: "The output will always be a SparseArray." However, when passing a non-SparseDtype (like `np.float64`), the method converts to dense and calls `astype_array`, which returns a regular ndarray (lines 1300-1305).

This violates the API contract and breaks code that expects `astype()` to always return a SparseArray.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1299,9 +1299,12 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):

         future_dtype = pandas_dtype(dtype)
         if not isinstance(future_dtype, SparseDtype):
-            # GH#34457
-            values = np.asarray(self)
-            values = ensure_wrapped_if_datetimelike(values)
-            return astype_array(values, dtype=future_dtype, copy=False)
+            # Convert to the requested dtype while maintaining sparse structure
+            # Wrap the dtype in SparseDtype to maintain sparseness
+            sparse_dtype = SparseDtype(future_dtype, fill_value=self.fill_value)
+            values = ensure_wrapped_if_datetimelike(self.sp_values)
+            sp_values = astype_array(values, future_dtype, copy=copy)
+            sp_values = np.asarray(sp_values)
+            return self._simple_new(sp_values, self.sp_index, sparse_dtype)

         dtype = self.dtype.update_dtype(dtype)
```