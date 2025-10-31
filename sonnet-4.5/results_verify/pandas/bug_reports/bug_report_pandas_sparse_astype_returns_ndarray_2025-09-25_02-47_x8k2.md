# Bug Report: pandas.core.sparse.SparseArray astype() Returns ndarray

**Target**: `pandas.core.arrays.sparse.array.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `astype()` method on SparseArray returns a numpy ndarray instead of a SparseArray when the target dtype is a plain numpy dtype (not a SparseDtype), violating its documented contract.

## Property-Based Test

```python
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st
import numpy as np

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50))
def test_sparse_array_astype_roundtrip(values):
    arr = SparseArray(values, dtype=np.int64)
    arr_float = arr.astype(np.float64)
    arr_int = arr_float.astype(np.int64)
    assert np.array_equal(arr.to_dense(), arr_int.to_dense()), \
        f"astype roundtrip failed: {arr.to_dense()} != {arr_int.to_dense()}"
```

**Failing input**: `[1, 2, 3]` (or any integer list)

## Reproducing the Bug

```python
from pandas.arrays import SparseArray
import numpy as np

arr = SparseArray([1, 2, 3], dtype=np.int64)
result = arr.astype(np.float64)

print(f'Type: {type(result)}')
print(f'Expected: <class pandas.core.arrays.sparse.array.SparseArray>')
print(f'Actual: {type(result)}')
```

Expected: `SparseArray([1.0, 2.0, 3.0], ...)`
Actual: `numpy.ndarray([1., 2., 3.])`

## Why This Is A Bug

The docstring for `astype()` explicitly states:

> "The output will always be a SparseArray. To convert to a dense ndarray with a certain dtype, use :meth:`numpy.asarray`."

However, when calling `astype()` with a plain numpy dtype (like `np.float64`), the method returns a numpy ndarray, not a SparseArray.

The implementation at lines 65-69 shows:

```python
if not isinstance(future_dtype, SparseDtype):
    # GH#34457
    values = np.asarray(self)
    values = ensure_wrapped_if_datetimelike(values)
    return astype_array(values, dtype=future_dtype, copy=False)
```

This code path returns the result of `astype_array()`, which is an ndarray, violating the documented contract.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -64,7 +64,11 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         future_dtype = pandas_dtype(dtype)
         if not isinstance(future_dtype, SparseDtype):
             # GH#34457
-            values = np.asarray(self)
-            values = ensure_wrapped_if_datetimelike(values)
-            return astype_array(values, dtype=future_dtype, copy=False)
+            # Convert to the new dtype but maintain SparseArray structure
+            # by wrapping in a SparseDtype
+            sparse_dtype = SparseDtype(future_dtype, fill_value=self.fill_value)
+            return self.astype(sparse_dtype, copy=copy)

         dtype = self.dtype.update_dtype(dtype)
```

Alternatively, the documentation could be updated to clarify that passing a non-SparseDtype returns a dense array, though this would be less intuitive.