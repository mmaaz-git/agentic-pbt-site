# Bug Report: numpy.ctypeslib.ndpointer Negative Shape Dimensions

**Target**: `numpy.ctypeslib.ndpointer`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ndpointer` function accepts negative values in `shape` parameter without validation, leading to confusing error messages during array validation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy.ctypeslib
import numpy as np


@given(st.tuples(st.integers(min_value=-10, max_value=-1), st.integers(min_value=1, max_value=10)))
@settings(max_examples=200)
def test_ndpointer_negative_shape(shape):
    try:
        ptr = numpy.ctypeslib.ndpointer(shape=shape)
        assert False, f"Should reject shape with negative dimensions {shape}"
    except (TypeError, ValueError):
        pass
```

**Failing input**: `shape=(-1, 3)` or `shape=(0, -1)`

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib

ptr = numpy.ctypeslib.ndpointer(shape=(-1, 3))
print(f"Created: {ptr}")
print(f"shape: {ptr._shape_}")

arr = np.zeros((2, 3))
try:
    ptr.from_param(arr)
except TypeError as e:
    print(f"Error: {e}")
```

Output:
```
Created: <class 'numpy.ctypeslib._ctypeslib.ndpointer_any_-1x3'>
shape: (-1, 3)
Error: array must have shape (-1, 3)
```

The pointer is created successfully but produces a nonsensical error message when used. Shape dimensions cannot be negative in NumPy.

## Why This Is A Bug

1. Negative shape dimensions are invalid - NumPy arrays cannot have negative dimensions
2. The function should validate the shape parameter at creation time
3. The resulting error message "array must have shape (-1, 3)" is misleading
4. This violates the principle of early validation - similar to the negative ndim issue

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -313,6 +313,9 @@ def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
     if shape is not None:
         try:
             shape = tuple(shape)
+            if any(s < 0 for s in shape):
+                raise ValueError(f"shape dimensions must be non-negative, got {shape}")
         except TypeError:
             # single integer -> 1-tuple
             shape = (shape,)
```