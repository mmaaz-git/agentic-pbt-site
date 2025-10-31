# Bug Report: numpy.ma.mask_or Crashes on List Inputs

**Target**: `numpy.ma.mask_or`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.ma.mask_or` crashes with `AttributeError` when given Python lists as inputs, despite its docstring explicitly stating it accepts "array_like" inputs.

## Property-Based Test

```python
import numpy.ma as ma
from hypothesis import given, settings, strategies as st, assume


@settings(max_examples=1000)
@given(st.lists(st.booleans(), min_size=1, max_size=20))
def test_mask_or_accepts_lists(mask1):
    assume(len(mask1) > 0)
    mask2 = [not m for m in mask1]
    result = ma.mask_or(mask1, mask2)
```

**Failing input**: `mask1=[False]`, `mask2=[True]`

## Reproducing the Bug

```python
import numpy.ma as ma

mask1 = [False, True, False]
mask2 = [True, False, False]

result = ma.mask_or(mask1, mask2)
```

**Output**:
```
AttributeError: 'NoneType' object has no attribute 'names'
```

## Why This Is A Bug

The docstring for `mask_or` explicitly states:

```
Parameters
----------
m1, m2 : array_like
    Input masks.
```

In NumPy, "array_like" conventionally includes Python lists, tuples, and other sequence types. The function works correctly with NumPy arrays but crashes with lists.

The error occurs at line 1808 in `numpy/ma/core.py` where the code tries to access `dtype1.names` without first checking if `dtype1` is None, which happens when the input is a Python list that hasn't been converted to an array.

Other mask utility functions like `make_mask` and `getmask` correctly handle list inputs, making this behavior inconsistent within the module.

## Fix

The function should convert array-like inputs to NumPy arrays before accessing their dtype attributes:

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -1800,6 +1800,8 @@ def mask_or(m1, m2, copy=False, shrink=True):
     The result may be a view on `m1` or `m2` if the other is `nomask`
     (i.e. False).
     """
+    m1 = np.asarray(m1)
+    m2 = np.asarray(m2)
     dtype1 = getattr(m1, 'dtype', None)
     dtype2 = getattr(m2, 'dtype', None)
     if dtype1 != dtype2:
```