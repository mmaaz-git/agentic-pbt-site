# Bug Report: pandas.core.arrays.sparse.SparseArray.astype() loses data when changing fill_value

**Target**: `pandas.core.arrays.sparse.SparseArray.astype()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When calling `SparseArray.astype()` with a `SparseDtype` that has a different `fill_value` than the original array, the actual values in the array are lost and replaced with the new fill_value. This violates the fundamental contract of `astype()` which should preserve array values while changing only the dtype.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import SparseArray

@given(
    data=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    fill_value1=st.integers(min_value=-1000, max_value=1000),
    fill_value2=st.integers(min_value=-1000, max_value=1000)
)
def test_sparse_array_astype_preserves_values(data, fill_value1, fill_value2):
    arr = np.array(data)
    sparse = SparseArray(arr, fill_value=fill_value1)

    dtype = pd.SparseDtype(np.float64, fill_value2)
    sparse_casted = sparse.astype(dtype)

    np.testing.assert_allclose(sparse_casted.to_dense(), arr.astype(np.float64))
```

**Failing input**: `data=[0], fill_value1=0, fill_value2=1`

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import SparseArray

data = [0]
sparse = SparseArray(data, fill_value=0)
print("Original:", sparse.to_dense())

dtype = pd.SparseDtype(np.float64, fill_value=1)
casted = sparse.astype(dtype)
print("After astype:", casted.to_dense())
print("Expected:    [0.]")
print("Actual:      [1.]")
```

Output:
```
Original: [0]
After astype: [1.]
Expected:    [0.]
Actual:      [1.]
```

## Why This Is A Bug

The `astype()` method is expected to convert the dtype of an array while preserving all values. However, when a SparseArray contains only fill_value elements and is cast to a dtype with a different fill_value, the actual values are lost.

This happens because:
1. When all array values equal the fill_value, `sp_values` is empty and `sp_index.indices` is empty (no non-fill values to store)
2. When `astype()` is called with a new fill_value, it creates a new SparseArray with the empty `sp_values` and empty `sp_index.indices`
3. The `to_dense()` method then fills all positions with the NEW fill_value instead of the ORIGINAL values

This violates the fundamental invariant: `array.astype(dtype).to_dense()` should equal `array.to_dense().astype(dtype)`.

## Fix

The issue is in the `astype` method at lines 1307-1314 of `/pandas/core/arrays/sparse/array.py`. When the fill_value changes, the method needs to handle values that were previously stored as "fill" but now need to be explicitly stored.

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1305,6 +1305,14 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
             return astype_array(values, dtype=future_dtype, copy=False)

         dtype = self.dtype.update_dtype(dtype)
+
+        # If fill_value is changing, we need to convert to dense first
+        # to preserve actual values, then convert back to sparse
+        if not self._fill_value_matches(dtype.fill_value):
+            dense_values = np.asarray(self)
+            dense_values = ensure_wrapped_if_datetimelike(dense_values)
+            dense_casted = astype_array(dense_values, dtype.subtype, copy=False)
+            return type(self)(dense_casted, fill_value=dtype.fill_value)
+
         subtype = pandas_dtype(dtype._subtype_with_str)
         subtype = cast(np.dtype, subtype)  # ensured by update_dtype
         values = ensure_wrapped_if_datetimelike(self.sp_values)