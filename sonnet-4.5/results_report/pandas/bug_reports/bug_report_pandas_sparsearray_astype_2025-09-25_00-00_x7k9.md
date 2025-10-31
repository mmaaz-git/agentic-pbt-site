# Bug Report: pandas.core.sparse.SparseArray.astype() Contract Violation

**Target**: `pandas.core.sparse.api.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `astype()` method's docstring explicitly states "The output will always be a SparseArray", but when converting to a non-SparseDtype (e.g., `np.float64`), it returns a numpy ndarray instead.

## Property-Based Test

```python
import numpy as np
from pandas.core.sparse.api import SparseArray
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100))
def test_astype_returns_sparsearray(data):
    sparse = SparseArray(data, dtype=np.int64)
    result = sparse.astype(np.float64)
    assert isinstance(result, SparseArray), \
        f"astype() should return SparseArray, got {type(result)}"
```

**Failing input**: `[0]` (or any list)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.sparse.api import SparseArray

sparse = SparseArray([1, 2, 3], dtype=np.int64)
result = sparse.astype(np.float64)

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

The docstring for `astype()` explicitly states:

> The output will always be a SparseArray. To convert to a dense ndarray with a certain dtype, use :meth:`numpy.asarray`.

However, when the target dtype is not a `SparseDtype` (e.g., `np.float64`, `np.int32`), the method returns a numpy ndarray. This violates the documented contract and breaks user expectations. Users who rely on the return type being a SparseArray will encounter unexpected errors.

The bug is in lines around 716-720 in `array.py`:

```python
if not isinstance(future_dtype, SparseDtype):
    values = np.asarray(self)
    values = ensure_wrapped_if_datetimelike(values)
    return astype_array(values, dtype=future_dtype, copy=False)
```

The `astype_array()` call returns a numpy array, not a SparseArray.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -714,7 +714,10 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         future_dtype = pandas_dtype(dtype)
         if not isinstance(future_dtype, SparseDtype):
             # GH#34457
             values = np.asarray(self)
             values = ensure_wrapped_if_datetimelike(values)
-            return astype_array(values, dtype=future_dtype, copy=False)
+            converted = astype_array(values, dtype=future_dtype, copy=False)
+            # Wrap back in SparseArray to honor the docstring contract
+            return SparseArray(converted)

         dtype = self.dtype.update_dtype(dtype)
         subtype = pandas_dtype(dtype._subtype_with_str)
```

The fix wraps the converted array back into a SparseArray before returning, ensuring the method honors its documented contract.