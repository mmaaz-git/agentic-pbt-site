# Bug Report: pandas.core.sparse.SparseArray.astype Returns Wrong Type

**Target**: `pandas.core.sparse.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `astype()` method of `SparseArray` violates its documented contract by returning a numpy ndarray instead of a SparseArray when converting to non-SparseDtype types.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.arrays import SparseArray
import numpy as np

@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100))
def test_astype_preserves_values(data):
    sparse = SparseArray(data, dtype=np.int64)
    sparse_float = sparse.astype(np.float64)

    assert isinstance(sparse_float, SparseArray)
    assert np.array_equal(sparse.to_dense(), sparse_float.to_dense())
```

**Failing input**: `[0]` (or any integer list)

## Reproducing the Bug

```python
from pandas.arrays import SparseArray
import numpy as np

sparse = SparseArray([1, 0, 0, 2], dtype=np.int64)
print(f"Original type: {type(sparse)}")

sparse_float = sparse.astype(np.float64)
print(f"After astype type: {type(sparse_float)}")
print(f"Is SparseArray: {isinstance(sparse_float, SparseArray)}")
```

Output:
```
Original type: <class 'pandas.core.arrays.sparse.array.SparseArray'>
After astype type: <class 'numpy.ndarray'>
Is SparseArray: False
```

## Why This Is A Bug

The docstring for `astype()` explicitly states: "The output will always be a SparseArray." However, when converting to a non-SparseDtype (e.g., `np.float64`), the method returns a numpy ndarray instead.

This violates the documented API contract and breaks code that expects `astype()` to preserve the array type.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -632,7 +632,7 @@ class SparseArray(OpsMixin, ExtensionArray):
         if not isinstance(future_dtype, SparseDtype):
             # GH#34457
             values = np.asarray(self)
             values = ensure_wrapped_if_datetimelike(values)
-            return astype_array(values, dtype=future_dtype, copy=False)
+            return type(self)(astype_array(values, dtype=future_dtype, copy=False))

         dtype = self.dtype.update_dtype(dtype)
```

The fix wraps the result in a SparseArray to match the documented behavior.