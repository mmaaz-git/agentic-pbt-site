# Bug Report: pandas.core.arrays.sparse.SparseArray.nonzero Missing Nonzero Fill Values

**Target**: `pandas.core.arrays.sparse.SparseArray.nonzero`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `SparseArray.nonzero()` method fails to count positions filled with a nonzero fill_value, returning inconsistent results compared to `to_dense().nonzero()`.

## Property-Based Test

```python
import numpy as np
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st

@given(
    st.lists(st.integers(min_value=-10, max_value=10), min_size=1, max_size=20),
    st.integers(min_value=-10, max_value=10)
)
def test_nonzero_matches_dense(data, fill_value):
    arr = SparseArray(data, fill_value=fill_value)
    sparse_result = arr.nonzero()[0]
    dense_result = arr.to_dense().nonzero()[0]

    assert np.array_equal(sparse_result, dense_result), \
        f"sparse.nonzero() != to_dense().nonzero()"
```

**Failing input**: `SparseArray([0, 1, 2, 2], fill_value=2)`

## Reproducing the Bug

```python
import numpy as np
from pandas.arrays import SparseArray

arr = SparseArray([0, 1, 2, 2], fill_value=2)

print(f"Array: {arr}")
print(f"to_dense(): {arr.to_dense()}")
print(f"Expected nonzero positions: {arr.to_dense().nonzero()[0]}")
print(f"Actual nonzero positions: {arr.nonzero()[0]}")

assert np.array_equal(arr.nonzero()[0], arr.to_dense().nonzero()[0])
```

Output:
```
Array: [0, 1, 2, 2]
Fill: 2
to_dense(): [0 1 2 2]
Expected nonzero positions: [1 2 3]
Actual nonzero positions: [1]
AssertionError
```

## Why This Is A Bug

The dense representation `[0, 1, 2, 2]` has nonzero values at positions `[1, 2, 3]`. However, `sparse.nonzero()` only returns `[1]`, missing positions `[2, 3]` which contain the nonzero fill_value `2`.

The current implementation is:

```python
def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
    if self.fill_value == 0:
        return (self.sp_index.indices,)
    else:
        return (self.sp_index.indices[self.sp_values != 0],)
```

When `fill_value != 0`, the method only considers explicitly stored values (`sp_values`), ignoring positions filled with the nonzero `fill_value`. This creates an inconsistency where `arr.nonzero()` doesn't match `arr.to_dense().nonzero()`.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -xxx,x +xxx,x @@ class SparseArray(OpsMixin, ExtensionArray):
     def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
         if self.fill_value == 0:
             return (self.sp_index.indices,)
         else:
-            return (self.sp_index.indices[self.sp_values != 0],)
+            # Need to account for positions with nonzero fill_value
+            nz_stored = self.sp_index.indices[self.sp_values != 0]
+            if self.fill_value != 0:
+                # Positions NOT in sp_index have fill_value (nonzero)
+                fill_positions = np.setdiff1d(np.arange(len(self)), self.sp_index.indices)
+                return (np.sort(np.concatenate([nz_stored, fill_positions])),)
+            return (nz_stored,)
```

Alternatively, a simpler fix that always computes from dense representation for correctness:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -xxx,x +xxx,x @@ class SparseArray(OpsMixin, ExtensionArray):
     def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
-        if self.fill_value == 0:
-            return (self.sp_index.indices,)
-        else:
-            return (self.sp_index.indices[self.sp_values != 0],)
+        return self.to_dense().nonzero()
```