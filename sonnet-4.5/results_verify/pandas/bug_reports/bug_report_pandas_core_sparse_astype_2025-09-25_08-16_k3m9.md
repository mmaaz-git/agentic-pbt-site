# Bug Report: pandas.core.sparse.SparseArray.astype Contract Violation

**Target**: `pandas.core.arrays.sparse.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `SparseArray.astype()` method violates its documented contract by returning a regular numpy array instead of a SparseArray when converting to a non-SparseDtype.

## Property-Based Test

```python
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st
import numpy as np


@given(st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=10))
def test_astype_returns_sparse_array(data):
    sparse_int = SparseArray(data)
    result = sparse_int.astype(np.float64)

    assert isinstance(result, SparseArray), \
        f"astype() docstring says 'The output will always be a SparseArray', but got {type(result)}"
```

**Failing input**: `[0]`

## Reproducing the Bug

```python
from pandas.arrays import SparseArray
import numpy as np

sparse_int = SparseArray([0, 0, 1, 2])
result = sparse_int.astype(np.float64)

print(f"Type of result: {type(result)}")
print(f"Is SparseArray: {isinstance(result, SparseArray)}")

assert isinstance(result, SparseArray)
```

Output:
```
Type of result: <class 'numpy.ndarray'>
Is SparseArray: False
AssertionError
```

## Why This Is A Bug

The docstring at line 1241-1242 of `pandas/core/arrays/sparse/array.py` explicitly states:

> "The output will always be a SparseArray. To convert to a dense ndarray with a certain dtype, use :meth:`numpy.asarray`."

However, the implementation at lines 1300-1305 shows:

```python
future_dtype = pandas_dtype(dtype)
if not isinstance(future_dtype, SparseDtype):
    values = np.asarray(self)
    values = ensure_wrapped_if_datetimelike(values)
    return astype_array(values, dtype=future_dtype, copy=False)
```

When the target dtype is not a SparseDtype (e.g., `np.float64`), the method returns the result of `astype_array()`, which is a regular numpy ndarray, not a SparseArray.

This violates the documented API contract and can break user code that expects a SparseArray return type.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1299,10 +1299,13 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):

         future_dtype = pandas_dtype(dtype)
         if not isinstance(future_dtype, SparseDtype):
-            # GH#34457
-            values = np.asarray(self)
-            values = ensure_wrapped_if_datetimelike(values)
-            return astype_array(values, dtype=future_dtype, copy=False)
+            # Convert to SparseDtype with same fill_value to maintain sparsity
+            future_dtype = SparseDtype(future_dtype, fill_value=self.fill_value)
+
+        # Now future_dtype is guaranteed to be a SparseDtype
+        if future_dtype == self._dtype:
+            return self.copy() if copy else self

         dtype = self.dtype.update_dtype(dtype)
         subtype = pandas_dtype(dtype._subtype_with_str)
```

Alternatively, if the current behavior is intended, the docstring should be corrected to:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1238,8 +1238,9 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         """
         Change the dtype of a SparseArray.

-        The output will always be a SparseArray. To convert to a dense
-        ndarray with a certain dtype, use :meth:`numpy.asarray`.
+        When converting to a SparseDtype, returns a SparseArray. When converting
+        to a regular numpy dtype, returns a regular numpy ndarray. To explicitly
+        convert to a dense ndarray, use :meth:`numpy.asarray`.
```