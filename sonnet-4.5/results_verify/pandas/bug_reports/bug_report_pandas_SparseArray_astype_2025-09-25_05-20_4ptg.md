# Bug Report: SparseArray.astype() Returns Dense Array Instead of SparseArray

**Target**: `pandas.core.arrays.sparse.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`SparseArray.astype()` violates its documented contract by returning a dense numpy array instead of a SparseArray when given a numpy dtype (e.g., `np.float64`). The documentation explicitly states "The output will always be a SparseArray" but this is not the case.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.core.arrays.sparse import SparseArray

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=10))
def test_astype_returns_sparse_array(data):
    arr = np.array(data, dtype=np.int64)
    sparse = SparseArray(arr)
    result = sparse.astype(np.float64)

    assert isinstance(result, SparseArray), \
        f"astype should return SparseArray per documentation, got {type(result).__name__}"
```

**Failing input**: `[0]` (or any other list)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.arrays.sparse import SparseArray

arr = SparseArray([0, 1, 2])
result = arr.astype(np.float64)

print(f"Type: {type(result)}")
assert isinstance(result, SparseArray), \
    f"Expected SparseArray, got {type(result).__name__}"
```

Output:
```
Type: <class 'numpy.ndarray'>
AssertionError: Expected SparseArray, got ndarray
```

## Why This Is A Bug

The `SparseArray.astype()` documentation explicitly states:

> "The output will always be a SparseArray. To convert to a dense ndarray with a certain dtype, use :meth:`numpy.asarray`."

However, when calling `astype(np.float64)` or `astype('float64')` with a numpy dtype directly (not wrapped in `SparseDtype`), the method returns a dense numpy array, violating this documented contract.

This is a significant API contract violation because:
1. Users relying on the documentation will expect a SparseArray and may encounter runtime errors
2. The sparse representation is lost, defeating the purpose of using SparseArray
3. It breaks the consistency of the ExtensionArray API where `astype` should preserve the array type

## Fix

The bug is in the `astype` method at lines handling non-SparseDtype dtypes:

```python
if not isinstance(future_dtype, SparseDtype):
    # GH#34457
    values = np.asarray(self)
    values = ensure_wrapped_if_datetimelike(values)
    return astype_array(values, dtype=future_dtype, copy=False)  # Returns ndarray
```

The fix should wrap the non-SparseDtype case in a SparseArray:

```diff
     if not isinstance(future_dtype, SparseDtype):
         # GH#34457
-        values = np.asarray(self)
-        values = ensure_wrapped_if_datetimelike(values)
-        return astype_array(values, dtype=future_dtype, copy=False)
+        # Convert to the new dtype while preserving sparse structure
+        sparse_dtype = SparseDtype(future_dtype, fill_value=self.fill_value)
+        return self.astype(sparse_dtype, copy=copy)
```

This ensures that when a user passes a numpy dtype, it's automatically wrapped in a SparseDtype with the current fill_value, preserving the sparse structure while changing the data type as requested.