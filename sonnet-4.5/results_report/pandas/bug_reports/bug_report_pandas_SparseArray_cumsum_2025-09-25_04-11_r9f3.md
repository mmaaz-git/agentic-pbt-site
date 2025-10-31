# Bug Report: pandas.arrays.SparseArray.cumsum Infinite Recursion

**Target**: `pandas.arrays.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Calling `cumsum()` on any `SparseArray` causes infinite recursion and crashes with `RecursionError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.arrays

@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=30))
@settings(max_examples=500)
def test_sparse_array_cumsum_length(data):
    sparse = pandas.arrays.SparseArray(data, fill_value=0)
    cumsum_result = sparse.cumsum()
    assert len(cumsum_result) == len(sparse)
```

**Failing input**: `data=[1]` (or any non-empty list)

## Reproducing the Bug

```python
import pandas.arrays

sparse = pandas.arrays.SparseArray([1, 0, 2, 0, 3], fill_value=0)
result = sparse.cumsum()
```

## Why This Is A Bug

The implementation of `SparseArray.cumsum()` at line 1550 of `pandas/core/arrays/sparse/array.py` is:

```python
def cumsum(self, *args, **kwargs):
    return SparseArray(self.to_dense()).cumsum()
```

This creates infinite recursion:
1. `SparseArray.cumsum()` converts to dense array
2. Wraps result in `SparseArray(...)`
3. Calls `.cumsum()` on the new SparseArray
4. This repeats indefinitely until RecursionError

The method should call `cumsum()` on the dense array BEFORE wrapping it back in SparseArray, or call the parent class method.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, ExtensionArray):
         -------
         SparseArray
         """
-        return SparseArray(self.to_dense()).cumsum()
+        return SparseArray(self.to_dense().cumsum())
```