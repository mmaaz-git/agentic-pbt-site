# Bug Report: pandas.core.sparse.SparseArray argmin/argmax Crash on All Fill Values

**Target**: `pandas.core.arrays.sparse.array.SparseArray.argmin` and `pandas.core.arrays.sparse.array.SparseArray.argmax`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `argmin()` and `argmax()` methods crash with `ValueError` when all elements in the SparseArray equal the fill value (i.e., when `npoints == 0`), instead of returning a valid index.

## Property-Based Test

```python
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=50)
)
@settings(max_examples=500)
def test_argmin_argmax_values(data):
    arr = SparseArray(data)

    argmin_idx = arr.argmin()
    argmax_idx = arr.argmax()

    assert arr[argmin_idx] == arr.min()
    assert arr[argmax_idx] == arr.max()
```

**Failing input**: `[0, 0]` (or any list where all values equal the fill value)

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

arr = SparseArray([0, 0, 0], fill_value=0)
arr.argmin()
```

Output:
```
ValueError: attempt to get argmin of an empty sequence
```

The same error occurs with `argmax()`:
```python
arr.argmax()
```

## Why This Is A Bug

The methods `argmin()` and `argmax()` should return the index of the minimum/maximum element, consistent with numpy's behavior. For an array like `[0, 0, 0]`, `argmin()` should return `0` (the first index), and `argmax()` should also return `0`.

The root cause is in lines 1654-1658 of `pandas/core/arrays/sparse/array.py`:

```python
idx = np.arange(values.shape[0])
non_nans = values[~mask]
non_nan_idx = idx[~mask]

_candidate = non_nan_idx[func(non_nans)]
```

When all elements equal the fill value, `self._sparse_values` is empty (because sparse values only stores non-fill values). This makes `non_nans` an empty array, and calling `np.argmin(empty_array)` raises a `ValueError`.

## Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1653,6 +1653,10 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):

     idx = np.arange(values.shape[0])
     non_nans = values[~mask]
     non_nan_idx = idx[~mask]
+
+    if len(non_nans) == 0:
+        # All values are fill_value or NaN, return first fill location
+        return self._first_fill_value_loc() if self._first_fill_value_loc() >= 0 else 0

     _candidate = non_nan_idx[func(non_nans)]
     candidate = index[_candidate]
```

The fix checks if there are no sparse values before attempting to find argmin/argmax. If all values are the fill value, it returns the location of the first fill value (or 0 if that fails).