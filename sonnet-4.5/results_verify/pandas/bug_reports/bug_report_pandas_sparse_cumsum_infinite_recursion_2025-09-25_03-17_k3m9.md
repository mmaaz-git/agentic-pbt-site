# Bug Report: SparseArray cumsum Infinite Recursion

**Target**: `pandas.core.arrays.sparse.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

SparseArray.cumsum() causes infinite recursion for arrays with non-null fill values (e.g., fill_value=0), leading to RecursionError.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.arrays.sparse import SparseArray

@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100))
def test_cumsum_matches_dense(values):
    sparse_arr = SparseArray(values, fill_value=0)
    sparse_cumsum = sparse_arr.cumsum().to_dense()
    dense_cumsum = sparse_arr.to_dense().cumsum()

    assert np.array_equal(sparse_cumsum, dense_cumsum), \
        "cumsum() on sparse should match cumsum() on dense"
```

**Failing input**: `[1, 2, 3]` (or any non-empty integer array with fill_value=0)

## Reproducing the Bug

```python
from pandas.core.arrays.sparse import SparseArray

sparse_arr = SparseArray([1, 2, 3], fill_value=0)
print(f"Sparse array: {sparse_arr.to_dense()}")

try:
    result = sparse_arr.cumsum()
    print(f"Success: {result.to_dense()}")
except RecursionError as e:
    print(f"RecursionError: {e}")
```

Output:
```
Sparse array: [1 2 3]
RecursionError: maximum recursion depth exceeded
```

## Why This Is A Bug

The bug is in line 1550 of `/pandas/core/arrays/sparse/array.py`:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()
```

This line creates a new SparseArray from `self.to_dense()` and then calls `.cumsum()` on it. However, the newly created SparseArray **also** has `_null_fill_value == False` (since it inherits the same fill_value=0), causing it to enter the same code path again, leading to infinite recursion.

The infinite recursion happens because:
1. `SparseArray([1,2,3], fill_value=0)` has `_null_fill_value == False`
2. Calling `.cumsum()` enters the `if not self._null_fill_value:` branch
3. It creates `SparseArray(self.to_dense())` which is `SparseArray([1,2,3])`
4. This new SparseArray also has `fill_value=0` and `_null_fill_value == False`
5. Calling `.cumsum()` on it enters the same branch again → infinite recursion

The fix is trivial: call `.cumsum()` on the **dense array** instead of on a new SparseArray.

## Fix

```diff
diff --git a/pandas/core/arrays/sparse/array.py b/pandas/core/arrays/sparse/array.py
index 1234567..abcdefg 100644
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         if axis is not None and axis >= self.ndim:
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return SparseArray(self.to_dense().cumsum())

         return SparseArray(
             self.sp_values.cumsum(),
```