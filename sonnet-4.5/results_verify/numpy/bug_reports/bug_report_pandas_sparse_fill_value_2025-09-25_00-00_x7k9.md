# Bug Report: SparseArray fill_value setter corrupts array values

**Target**: `pandas.core.arrays.sparse.SparseArray.fill_value` (setter)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Setting the `fill_value` property on a `SparseArray` corrupts the logical array values. The setter only updates the dtype's fill_value but does not update the sparse representation, causing `to_dense()` to return incorrect values.

## Property-Based Test

```python
from pandas.arrays import SparseArray
import numpy as np
from hypothesis import given, strategies as st

@given(
    st.lists(st.integers(min_value=-10, max_value=10), min_size=3, max_size=20),
    st.integers(min_value=-10, max_value=10),
    st.integers(min_value=-10, max_value=10)
)
def test_fill_value_setter_preserves_dense_values(values, old_fill, new_fill):
    arr = SparseArray(values, fill_value=old_fill)
    original_dense = arr.to_dense().copy()

    arr.fill_value = new_fill
    new_dense = arr.to_dense()

    np.testing.assert_array_equal(new_dense, original_dense,
        err_msg=f"Changing fill_value from {old_fill} to {new_fill} changed array values")
```

**Failing input**: `values=[0, 0, 1, 2], old_fill=0, new_fill=1`

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

arr = SparseArray([0, 0, 1, 2], fill_value=0)
print(arr.to_dense())

arr.fill_value = 1
print(arr.to_dense())
```

**Expected output:**
```
[0 0 1 2]
[0 0 1 2]
```

**Actual output:**
```
[0 0 1 2]
[1 1 1 2]
```

## Why This Is A Bug

The invariant that should hold is: **changing metadata (like `fill_value`) should not change the logical array values**. The `fill_value` is just an implementation detail for sparse storage efficiency - it should not affect what values the array logically contains.

When a user sets `arr.fill_value = new_value`, the logical array values should remain unchanged. However, the current implementation only updates the dtype without reconstructing the sparse representation, causing `to_dense()` to produce different values.

## Fix

The `fill_value` setter needs to reconstruct the sparse representation when the fill value changes. Here's a corrected implementation:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -658,7 +658,14 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):

     @fill_value.setter
     def fill_value(self, value) -> None:
-        self._dtype = SparseDtype(self.dtype.subtype, value)
+        if value != self.fill_value:
+            # Reconstruct the sparse array with the new fill value
+            # to maintain the same logical dense values
+            dense = self.to_dense()
+            new_sparse = type(self)(dense, fill_value=value, kind=self.kind)
+            self._sparse_values = new_sparse._sparse_values
+            self._sparse_index = new_sparse._sparse_index
+            self._dtype = new_sparse._dtype
+        else:
+            self._dtype = SparseDtype(self.dtype.subtype, value)
```

This ensures that the sparse representation is updated to maintain the same logical values when the fill value changes.