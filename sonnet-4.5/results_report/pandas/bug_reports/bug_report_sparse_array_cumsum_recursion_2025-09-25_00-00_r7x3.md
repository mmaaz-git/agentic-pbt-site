# Bug Report: SparseArray.cumsum Infinite Recursion

**Target**: `pandas.core.arrays.sparse.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method on `SparseArray` enters infinite recursion when called on arrays with non-null fill values (e.g., integer arrays with fill_value=0, or any boolean array), causing `RecursionError` and crashing the application.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.arrays import SparseArray

@st.composite
def sparse_arrays(draw, min_size=0, max_size=100):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    dtype_choice = draw(st.sampled_from(['int64', 'bool']))

    if dtype_choice == 'int64':
        values = draw(st.lists(st.integers(min_value=-1000, max_value=1000),
                              min_size=size, max_size=size))
        fill_value = 0
    else:
        values = draw(st.lists(st.booleans(), min_size=size, max_size=size))
        fill_value = False

    kind = draw(st.sampled_from(['integer', 'block']))
    return SparseArray(values, fill_value=fill_value, kind=kind)

@given(sparse_arrays())
@settings(max_examples=100)
def test_cumsum_length(arr):
    """Cumsum should preserve length"""
    cumsum_result = arr.cumsum()
    assert len(cumsum_result) == len(arr)
```

**Failing input**: `SparseArray([True], fill_value=False)`

## Reproducing the Bug

```python
from pandas.arrays import SparseArray
import sys

sys.setrecursionlimit(50)

arr = SparseArray([True], fill_value=False)
print(f"Array: {arr}")
print(f"_null_fill_value: {arr._null_fill_value}")

result = arr.cumsum()
```

**Output:**
```
Array: [True]
Fill: False
IntIndex
Indices: array([0], dtype=int32)

_null_fill_value: False
RecursionError: maximum recursion depth exceeded
```

## Why This Is A Bug

The implementation has a critical logic error. When `_null_fill_value` is False (meaning fill_value is not NaN/None), the code does:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()  # BUG HERE!
```

This creates infinite recursion because:
1. `self.to_dense()` returns a NumPy array
2. `SparseArray(...)` wraps it back into a SparseArray with the same non-null fill_value
3. `.cumsum()` is called on this new SparseArray
4. The condition `not self._null_fill_value` is still True
5. Go to step 1 → infinite loop

This affects all arrays where the fill_value cannot be NaN:
- All boolean sparse arrays (boolean dtype cannot have NaN)
- Integer sparse arrays with non-NaN fill values
- Float sparse arrays with explicit non-NaN fill values like 0.0

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         if axis is not None and axis >= self.ndim:  # Mimic ndarray behaviour.
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return SparseArray(self.to_dense().cumsum())

         return SparseArray(
             self.sp_values.cumsum(),
```

The fix calls `cumsum()` on the dense NumPy array before wrapping the result in a SparseArray, breaking the infinite recursion.