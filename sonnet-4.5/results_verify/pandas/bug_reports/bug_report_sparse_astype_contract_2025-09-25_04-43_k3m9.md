# Bug Report: SparseArray.astype() Contract Violation

**Target**: `pandas.core.arrays.sparse.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`SparseArray.astype()` violates its documented contract by returning a `numpy.ndarray` instead of a `SparseArray` when given a NumPy dtype (e.g., `np.int64`). The docstring explicitly states "The output will always be a SparseArray."

## Property-Based Test

```python
from pandas.core.arrays.sparse import SparseArray
import numpy as np
from hypothesis import given, strategies as st

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
             min_size=1, max_size=20)
)
def test_astype_returns_sparse_array(data):
    fill_value = data[0] if data else 0.0
    arr = SparseArray(data, fill_value=fill_value)

    result = arr.astype(np.int64)

    assert isinstance(result, SparseArray), \
        f"astype should return SparseArray but returned {type(result)}"
```

**Failing input**: `[0.0]`

## Reproducing the Bug

```python
from pandas.core.arrays.sparse import SparseArray
import numpy as np

arr = SparseArray([0.0, 1.0, 0.0], fill_value=0.0)
result = arr.astype(np.int64)

print(f"Result type: {type(result)}")
print(f"Is SparseArray? {isinstance(result, SparseArray)}")
```

**Output:**
```
Result type: <class 'numpy.ndarray'>
Is SparseArray? False
```

## Why This Is A Bug

The `astype` method's docstring explicitly states:

> "The output will **always** be a SparseArray. To convert to a dense ndarray with a certain dtype, use :meth:`numpy.asarray`."

Additionally, the Parameters section documents that `dtype` can be "np.dtype or ExtensionDtype", implying that NumPy dtypes should be supported while still returning a SparseArray.

However, when a plain NumPy dtype (not wrapped in `SparseDtype`) is passed, the method returns a `numpy.ndarray`, violating this contract.

The implementation shows that for non-SparseDtype inputs, it explicitly converts to a dense array:

```python
if not isinstance(future_dtype, SparseDtype):
    # GH#34457
    values = np.asarray(self)
    values = ensure_wrapped_if_datetimelike(values)
    return astype_array(values, dtype=future_dtype, copy=False)  # Returns ndarray!
```

## Fix

The method should wrap the result in a new SparseArray when returning, to maintain the documented contract:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -385,7 +385,10 @@ class SparseArray(OpsMixin, ExtensionArray):
         if not isinstance(future_dtype, SparseDtype):
             # GH#34457
             values = np.asarray(self)
             values = ensure_wrapped_if_datetimelike(values)
-            return astype_array(values, dtype=future_dtype, copy=False)
+            values = astype_array(values, dtype=future_dtype, copy=False)
+            # Maintain contract: always return SparseArray
+            sparse_dtype = SparseDtype(future_dtype, fill_value=self.fill_value)
+            return type(self)(values, dtype=sparse_dtype, copy=False)

         dtype = self.dtype.update_dtype(dtype)
         subtype = pandas_dtype(dtype._subtype_with_str)
```