# Bug Report: pandas.core.arrays.sparse.SparseArray.astype Contract Violation

**Target**: `pandas.core.arrays.sparse.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`SparseArray.astype()` violates its documented contract by returning a dense `numpy.ndarray` instead of a `SparseArray` when given a numpy dtype (e.g., `'float64'`) rather than a `SparseDtype`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.arrays import SparseArray

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100))
def test_astype_always_returns_sparse_array(values):
    arr = np.array(values)
    sparse = SparseArray(arr)

    result = sparse.astype('float64')

    assert isinstance(result, SparseArray), (
        f"astype() should always return a SparseArray according to its docstring, "
        f"but got {type(result)}"
    )
```

**Failing input**: `[1]` (or any list of integers)

## Reproducing the Bug

```python
import numpy as np
from pandas.arrays import SparseArray

sparse = SparseArray([1, 0, 0, 2])
result = sparse.astype('float64')

print(type(result))
```

**Output**: `<class 'numpy.ndarray'>`
**Expected**: `<class 'pandas.core.arrays.sparse.array.SparseArray'>`

## Why This Is A Bug

The `astype` method's docstring (lines 1238-1293 in `array.py`) explicitly states:

> "The output will always be a SparseArray. To convert to a dense ndarray with a certain dtype, use :meth:`numpy.asarray`."

The return type annotation also specifies `SparseArray` (line 1258).

However, the implementation (lines 1300-1305) contradicts this:

```python
future_dtype = pandas_dtype(dtype)
if not isinstance(future_dtype, SparseDtype):
    # GH#34457
    values = np.asarray(self)
    values = ensure_wrapped_if_datetimelike(values)
    return astype_array(values, dtype=future_dtype, copy=False)
```

When the dtype is not a `SparseDtype`, the code converts the sparse array to dense and returns the result from `astype_array()`, which is a dense array. This violates the documented guarantee.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1299,10 +1299,14 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):

         future_dtype = pandas_dtype(dtype)
         if not isinstance(future_dtype, SparseDtype):
-            # GH#34457
-            values = np.asarray(self)
-            values = ensure_wrapped_if_datetimelike(values)
-            return astype_array(values, dtype=future_dtype, copy=False)
+            # Convert numpy dtype to SparseDtype to maintain sparsity
+            # and honor the documented contract that astype always returns SparseArray
+            future_dtype = SparseDtype(future_dtype, fill_value=self.fill_value)
+            dtype = self.dtype.update_dtype(future_dtype)
+            subtype = pandas_dtype(dtype._subtype_with_str)
+            subtype = cast(np.dtype, subtype)
+            # Fall through to the SparseDtype handling below
+            # (lines 1307-1314 will handle the conversion)

         dtype = self.dtype.update_dtype(dtype)
         subtype = pandas_dtype(dtype._subtype_with_str)
```