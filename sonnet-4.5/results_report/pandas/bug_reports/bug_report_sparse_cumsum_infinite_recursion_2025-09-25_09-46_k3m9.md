# Bug Report: SparseArray cumsum() Infinite Recursion

**Target**: `pandas.core.arrays.sparse.array.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method on `SparseArray` with a non-NA fill value causes infinite recursion, leading to a `RecursionError` crash.

## Property-Based Test

```python
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st

@st.composite
def sparse_with_fill(draw, min_size=1, max_size=50):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    fill_value = draw(st.integers(min_value=-10, max_value=10))
    values = draw(st.lists(
        st.integers(min_value=-100, max_value=100),
        min_size=size, max_size=size
    ))
    return SparseArray(values, fill_value=fill_value)

@given(sparse_with_fill(min_size=5))
def test_cumsum_preserves_length(arr):
    cumsum = arr.cumsum()
    assert len(cumsum) == len(arr)
```

**Failing input**: Any `SparseArray` with a non-NA fill value (e.g., `fill_value=0`)

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

arr = SparseArray([1, 2, 3], fill_value=0)
result = arr.cumsum()
```

**Expected**: Returns a SparseArray with values `[1, 3, 6]`

**Actual**: Raises `RecursionError: maximum recursion depth exceeded`

## Why This Is A Bug

The `cumsum()` method is a documented feature of `SparseArray` and should work for all valid inputs. When the fill value is not NA, the code at line 1550 of `array.py` incorrectly creates a new `SparseArray` and calls `cumsum()` on it recursively:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()  # BUG: infinite recursion!
```

This creates an infinite loop because the newly created `SparseArray` will also have `_null_fill_value == False` and will recursively call `cumsum()` again.

The recursion stack looks like:
```
SparseArray([1,2,3]).cumsum()
  -> SparseArray([1,2,3]).cumsum()  # new SparseArray created, cumsum called
    -> SparseArray([1,2,3]).cumsum()  # new SparseArray created, cumsum called
      -> ... (infinite recursion)
```

## Fix

The fix is to call `cumsum()` on the dense NumPy array directly, not on a new `SparseArray`:

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
+            return type(self)(self.to_dense().cumsum())

         return SparseArray(
             self.sp_values.cumsum(),
```

This ensures that `cumsum()` is called on the NumPy array returned by `to_dense()`, avoiding the infinite recursion.