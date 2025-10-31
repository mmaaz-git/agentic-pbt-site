# Bug Report: SparseArray.astype Contract Violation

**Target**: `pandas.core.arrays.sparse.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `SparseArray.astype()` method's docstring states "The output will always be a SparseArray", but the implementation returns a dense numpy ndarray when given a non-SparseDtype argument.

## Property-Based Test

```python
@given(st.integers(min_value=1, max_value=50))
@settings(max_examples=300)
def test_astype_preserves_values(size):
    data = list(range(size))
    arr_int = SparseArray(data, fill_value=0)

    arr_float = arr_int.astype('float64')

    assert isinstance(arr_float, SparseArray), f"Expected SparseArray, got {type(arr_float)}"
```

**Failing input**: Any SparseArray with a non-SparseDtype astype argument (e.g., `'float64'`, `'int32'`, etc.)

## Reproducing the Bug

```python
from pandas.arrays import SparseArray
import numpy as np

arr = SparseArray([0, 1, 2, 0, 3], fill_value=0)
result = arr.astype('float64')

print(f"Type of result: {type(result)}")
print(f"Is SparseArray: {isinstance(result, SparseArray)}")
```

Output:
```
Type of result: <class 'numpy.ndarray'>
Is SparseArray: False
```

## Why This Is A Bug

The docstring at line 1241 of `pandas/core/arrays/sparse/array.py` explicitly states:

> "The output will always be a SparseArray. To convert to a dense ndarray with a certain dtype, use :meth:`numpy.asarray`."

However, when `dtype` is not a `SparseDtype` (e.g., just `'float64'` instead of `SparseDtype('float64')`), the code at lines 1300-1305 converts to dense and returns a numpy ndarray, violating the documented API contract.

This is a **contract violation**: the API documentation promises one behavior but delivers another.

## Fix

The implementation should be updated to always return a SparseArray, or the docstring should be corrected to reflect the actual behavior.

**Option 1**: Update the implementation to always return SparseArray:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1300,7 +1300,10 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         future_dtype = pandas_dtype(dtype)
         if not isinstance(future_dtype, SparseDtype):
             # GH#34457
-            values = np.asarray(self)
-            values = ensure_wrapped_if_datetimelike(values)
-            return astype_array(values, dtype=future_dtype, copy=False)
+            # Convert to SparseDtype to maintain sparse representation
+            sparse_dtype = SparseDtype(future_dtype, self.fill_value)
+            return self.astype(sparse_dtype, copy=copy)

         dtype = self.dtype.update_dtype(dtype)
```

**Option 2**: Update the docstring to reflect actual behavior:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1238,8 +1238,9 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         """
         Change the dtype of a SparseArray.

-        The output will always be a SparseArray. To convert to a dense
-        ndarray with a certain dtype, use :meth:`numpy.asarray`.
+        When `dtype` is a SparseDtype, returns a SparseArray. When `dtype` is
+        a regular numpy dtype, returns a dense ndarray. To explicitly convert
+        to dense, use :meth:`numpy.asarray`.

         Parameters
```

**Recommended**: Option 1 is preferred to maintain backward compatibility and honor the original design intent of always returning a SparseArray.