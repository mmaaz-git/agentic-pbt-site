# Bug Report: pandas.core.arrays.SparseArray Fill Value Change Corrupts Data

**Target**: `pandas.core.arrays.SparseArray`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When constructing a `SparseArray` from another `SparseArray` with a different `fill_value`, the actual data values are corrupted and replaced with the new fill value instead of being preserved.

## Property-Based Test

```python
import numpy as np
from pandas.core.arrays import SparseArray
from hypothesis import given, strategies as st, settings


@given(
    data=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=5, max_size=50),
    old_fill=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    new_fill=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=500)
def test_sparsearray_change_fill_value(data, old_fill, new_fill):
    sparse = SparseArray(data, fill_value=old_fill)
    original_dense = sparse.to_dense()

    new_sparse = SparseArray(sparse, fill_value=new_fill)

    assert np.allclose(new_sparse.to_dense(), original_dense, equal_nan=True, rtol=1e-10)
```

**Failing input**: `data=[0.0, 0.0, 0.0, 0.0, 0.0]`, `old_fill=0.0`, `new_fill=1.0`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.arrays import SparseArray

sparse = SparseArray([0.0, 0.0, 0.0, 0.0, 0.0], fill_value=0.0)
print(f"Original: {sparse.to_dense()}")

new_sparse = SparseArray(sparse, fill_value=1.0)
print(f"After changing fill_value to 1.0: {new_sparse.to_dense()}")

assert np.allclose(new_sparse.to_dense(), [0.0, 0.0, 0.0, 0.0, 0.0])
```

## Why This Is A Bug

When constructing a `SparseArray` from another `SparseArray`, the constructor reuses the sparse index and sparse values from the original array. However, when the `fill_value` is different, this is incorrect because:

1. Values equal to the old fill_value are not stored in `sp_values` (they're implicit)
2. When the fill_value changes, these implicit values become explicit and should be stored
3. The current implementation keeps the empty sparse index, so all values default to the new fill_value

In the example above, all five `0.0` values were implicitly stored (empty `sp_values`, empty index). When `fill_value` changes to `1.0`, those implicit values are now interpreted as `1.0` instead of `0.0`.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -374,12 +374,17 @@ class SparseArray(OpsMixin, ExtensionArray):
             fill_value = dtype.fill_value

         if isinstance(data, type(self)):
+            # If fill_value is changing, we need to convert to dense first
+            # to avoid data corruption
+            if fill_value is not None and fill_value != data.fill_value:
+                data = data.to_dense()
+            else:
             # disable normal inference on dtype, sparse_index, & fill_value
-            if sparse_index is None:
-                sparse_index = data.sp_index
-            if fill_value is None:
-                fill_value = data.fill_value
-            if dtype is None:
-                dtype = data.dtype
-            # TODO: make kind=None, and use data.kind?
-            data = data.sp_values
+                if sparse_index is None:
+                    sparse_index = data.sp_index
+                if fill_value is None:
+                    fill_value = data.fill_value
+                if dtype is None:
+                    dtype = data.dtype
+                # TODO: make kind=None, and use data.kind?
+                data = data.sp_values

         # Handle use-provided dtype
```