# Bug Report: pandas.core.arrays.sparse.SparseArray.nonzero Incorrect Logic

**Target**: `pandas.core.arrays.sparse.SparseArray.nonzero`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `nonzero()` method returns incorrect indices when the fill value is nonzero. It returns indices of non-fill values instead of indices of nonzero values, causing it to miss positions where the value equals a nonzero fill value.

## Property-Based Test

```python
import numpy as np
import pandas.core.arrays.sparse as sparse
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100),
    st.integers(min_value=-10, max_value=10)
)
@settings(max_examples=200)
def test_nonzero_equivalence(data, fill_value):
    """nonzero should match dense nonzero"""
    arr = sparse.SparseArray(data, fill_value=fill_value)

    sparse_nonzero = arr.nonzero()
    dense_nonzero = arr.to_dense().nonzero()

    for s, d in zip(sparse_nonzero, dense_nonzero):
        np.testing.assert_array_equal(
            s, d,
            err_msg="nonzero() mismatch"
        )
```

**Failing input**: `SparseArray([1], fill_value=1)` and `SparseArray([1, 2, 3], fill_value=2)`

## Reproducing the Bug

```python
import pandas.core.arrays.sparse as sparse
import numpy as np

arr1 = sparse.SparseArray([1], fill_value=1)
print(arr1.nonzero())
print(arr1.to_dense().nonzero())

arr2 = sparse.SparseArray([1, 2, 3], fill_value=2)
print(arr2.nonzero())
print(arr2.to_dense().nonzero())
```

Output:
```
(array([], dtype=int32),)
(array([0]),)

(array([0, 2], dtype=int32),)
(array([0, 1, 2]),)
```

## Why This Is A Bug

The `nonzero()` method is supposed to return indices where values are nonzero, following NumPy's convention. However, when the fill value is nonzero:

1. **Case 1**: Array `[1]` with `fill_value=1`
   - Returns: `[]` (empty)
   - Expected: `[0]` (because value 1 is nonzero)
   - Issue: Misses the position because 1 equals the fill value

2. **Case 2**: Array `[1, 2, 3]` with `fill_value=2`
   - Returns: `[0, 2]`
   - Expected: `[0, 1, 2]` (all values are nonzero)
   - Issue: Misses index 1 where value equals the nonzero fill value

The current implementation in `pandas/core/arrays/sparse/array.py`:

```python
def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
    if self.fill_value == 0:
        return (self.sp_index.indices,)
    else:
        return (self.sp_index.indices[self.sp_values != 0],)
```

This logic only checks sparse values for nonzero, but completely ignores positions with the fill value. When `fill_value` is nonzero (e.g., 1, 2, -1), those positions should be included in the result.

## Fix

The correct implementation needs to check all positions in the array:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -561,9 +561,9 @@ class SparseArray(OpsMixin, ExtensionArray):
     def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
-        if self.fill_value == 0:
-            return (self.sp_index.indices,)
-        else:
-            return (self.sp_index.indices[self.sp_values != 0],)
+        # Convert to dense to correctly handle nonzero fill values
+        # This ensures all nonzero positions are found, including those
+        # equal to a nonzero fill_value
+        return self.to_dense().nonzero()
```

Alternative optimization (if performance is critical):

```diff
--- a/pandas/core/arrays.sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -561,9 +561,13 @@ class SparseArray(OpsMixin, ExtensionArray):
     def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
         if self.fill_value == 0:
+            # Only sparse values are nonzero
             return (self.sp_index.indices,)
+        elif self.fill_value != 0:
+            # All positions (sparse + fill) may be nonzero
+            # Must check full array
+            return self.to_dense().nonzero()
         else:
-            return (self.sp_index.indices[self.sp_values != 0],)
+            # NaN fill value case
+            return (self.sp_index.indices[self.sp_values != 0],)
```

The simple fix calls `to_dense().nonzero()` which sacrifices the sparse optimization but guarantees correctness for all cases.