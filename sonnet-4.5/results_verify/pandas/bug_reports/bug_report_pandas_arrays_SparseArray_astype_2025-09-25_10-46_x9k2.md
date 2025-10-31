# Bug Report: SparseArray.astype() Returns ndarray Instead of SparseArray

**Target**: `pandas.arrays.SparseArray.astype()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

SparseArray.astype() docstring explicitly states "The output will always be a SparseArray", but when called with a NumPy dtype (e.g., 'int64'), it returns a numpy.ndarray instead.

## Property-Based Test

```python
import pandas.arrays as pa
from hypothesis import given, strategies as st, settings


@settings(max_examples=500)
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=50))
def test_sparse_astype_returns_sparse_array(values):
    sparse = pa.SparseArray(values, fill_value=0.0)
    result = sparse.astype('int64')
    assert isinstance(result, pa.SparseArray), f"astype() docstring says 'The output will always be a SparseArray' but got {type(result)}"
```

**Failing input**: `values=[0.0]`

## Reproducing the Bug

```python
import pandas.arrays as pa

sparse = pa.SparseArray([1.0, 2.0, 3.0, 0.0, 0.0], fill_value=0.0)
result = sparse.astype('int64')

print(f"Type: {type(result)}")
print(f"Expected: <class 'pandas.core.arrays.sparse.array.SparseArray'>")
print(f"Actual: {type(result)}")

assert isinstance(result, pa.SparseArray)
```

Output:
```
Type: <class 'numpy.ndarray'>
Expected: <class 'pandas.core.arrays.sparse.array.SparseArray'>
Actual: <class 'numpy.ndarray'>
AssertionError
```

## Why This Is A Bug

The docstring for `SparseArray.astype()` explicitly states: "The output will always be a SparseArray."

However, when calling `astype()` with a plain NumPy dtype string (like 'int64'), the method returns a numpy.ndarray instead of maintaining the SparseArray type. This violates the documented API contract.

## Fix

The documentation should be corrected to clarify that `astype()` with a plain NumPy dtype returns a dense ndarray, while `astype()` with a `SparseDtype` returns a `SparseArray`. Alternatively, the implementation could be changed to always return a SparseArray as documented.

Suggested documentation fix:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1,7 +1,8 @@
 """
 Change the dtype of a SparseArray.

-The output will always be a SparseArray. To convert to a dense
+The output will be a SparseArray if `dtype` is a SparseDtype,
+otherwise returns a dense ndarray. To explicitly convert to a dense
 ndarray with a certain dtype, use :meth:`numpy.asarray`.

 Parameters
```