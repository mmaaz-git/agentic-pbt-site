# Bug Report: numpy.ma.mask_or AttributeError with array_like inputs

**Target**: `numpy.ma.mask_or`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.ma.mask_or` crashes with `AttributeError: 'NoneType' object has no attribute 'names'` when passed plain Python lists, despite its docstring claiming to accept `array_like` inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy.ma as ma
import numpy as np


@given(
    m1=st.lists(st.booleans(), min_size=1, max_size=50),
    m2=st.lists(st.booleans(), min_size=1, max_size=50)
)
def test_mask_or_symmetry(m1, m2):
    assume(len(m1) == len(m2))

    result1 = ma.mask_or(m1, m2)
    result2 = ma.mask_or(m2, m1)

    if result1 is ma.nomask and result2 is ma.nomask:
        pass
    elif result1 is ma.nomask or result2 is ma.nomask:
        assert False, f"mask_or should be symmetric, but one is nomask: {result1} vs {result2}"
    else:
        assert np.array_equal(result1, result2), f"mask_or not symmetric: {result1} vs {result2}"
```

**Failing input**: `m1=[False], m2=[False]`

## Reproducing the Bug

```python
import numpy.ma as ma

m1 = [False]
m2 = [False]
result = ma.mask_or(m1, m2)
```

## Why This Is A Bug

The `mask_or` function's docstring explicitly states that `m1` and `m2` should be `array_like`, which in NumPy includes Python lists. However, the implementation crashes when given plain Python lists because it attempts to access `dtype.names` without first checking if `dtype` is `None` (which it is for Python lists).

The bug is at line 1808 in `/numpy/ma/core.py`:

```python
(dtype1, dtype2) = (getattr(m1, 'dtype', None), getattr(m2, 'dtype', None))
if dtype1 != dtype2:
    raise ValueError(f"Incompatible dtypes '{dtype1}'<>'{dtype2}'")
if dtype1.names is not None:  # BUG: dtype1 can be None!
```

## Fix

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -1805,7 +1805,7 @@ def mask_or(m1, m2, copy=False, shrink=True):
     (dtype1, dtype2) = (getattr(m1, 'dtype', None), getattr(m2, 'dtype', None))
     if dtype1 != dtype2:
         raise ValueError(f"Incompatible dtypes '{dtype1}'<>'{dtype2}'")
-    if dtype1.names is not None:
+    if dtype1 is not None and dtype1.names is not None:
         # Allocate an output mask array with the properly broadcast shape.
         newmask = np.empty(np.broadcast(m1, m2).shape, dtype1)
         _recursive_mask_or(m1, m2, newmask)
```