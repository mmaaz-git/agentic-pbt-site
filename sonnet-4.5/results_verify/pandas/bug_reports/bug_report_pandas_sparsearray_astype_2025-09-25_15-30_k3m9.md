# Bug Report: pandas.core.sparse.api.SparseArray.astype returns ndarray instead of SparseArray

**Target**: `pandas.core.sparse.api.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`SparseArray.astype()` violates its documented contract by returning a `numpy.ndarray` instead of a `SparseArray` when converting to a non-SparseDtype (e.g., `np.int64`, `np.float32`).

## Property-Based Test

```python
from pandas.core.sparse.api import SparseArray
import numpy as np
from hypothesis import given
import hypothesis.extra.numpy as npst


@given(npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=1, max_dims=1)))
def test_astype_preserves_type(arr):
    sparse = SparseArray(arr)
    result = sparse.astype(np.int64)
    assert isinstance(result, SparseArray), f"astype should return SparseArray, got {type(result)}"
```

**Failing input**: `array([0.])`

## Reproducing the Bug

```python
from pandas.core.sparse.api import SparseArray
import numpy as np

sparse = SparseArray([1.0, 0.0, 2.0])
result = sparse.astype(np.int64)

print(f"Type: {type(result)}")
print(f"Expected: <class 'pandas.core.arrays.sparse.array.SparseArray'>")
print(f"Actual: <class 'numpy.ndarray'>")

assert isinstance(result, SparseArray)
```

## Why This Is A Bug

The docstring for `SparseArray.astype()` explicitly states:

> "The output will always be a SparseArray. To convert to a dense ndarray with a certain dtype, use :meth:`numpy.asarray`."

However, when passing a non-SparseDtype (like `np.int64`), the method returns a `numpy.ndarray` instead of a `SparseArray`, violating this documented guarantee.

## Fix

The bug is in lines 1301-1305 of `/pandas/core/arrays/sparse/array.py`:

```python
if not isinstance(future_dtype, SparseDtype):
    # GH#34457
    values = np.asarray(self)
    values = ensure_wrapped_if_datetimelike(values)
    return astype_array(values, dtype=future_dtype, copy=False)
```

The method should wrap the result in a SparseArray before returning:

```diff
         if not isinstance(future_dtype, SparseDtype):
             # GH#34457
             values = np.asarray(self)
             values = ensure_wrapped_if_datetimelike(values)
-            return astype_array(values, dtype=future_dtype, copy=False)
+            result = astype_array(values, dtype=future_dtype, copy=False)
+            # Ensure we return a SparseArray as documented
+            future_dtype = SparseDtype(future_dtype)
+            return type(self)(result, dtype=future_dtype, copy=False)
```