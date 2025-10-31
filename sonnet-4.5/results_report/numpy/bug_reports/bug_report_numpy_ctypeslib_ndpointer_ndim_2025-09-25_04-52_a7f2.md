# Bug Report: numpy.ctypeslib.ndpointer Negative ndim

**Target**: `numpy.ctypeslib.ndpointer`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ndpointer` function accepts negative `ndim` values without validation, leading to confusing error messages during array validation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy.ctypeslib
import numpy as np


@given(st.integers(min_value=-1000, max_value=-1))
@settings(max_examples=200)
def test_ndpointer_negative_ndim(ndim):
    try:
        ptr = numpy.ctypeslib.ndpointer(ndim=ndim)
        assert False, f"Should reject negative ndim {ndim}"
    except (TypeError, ValueError):
        pass
```

**Failing input**: `ndim=-1`

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib

ptr = numpy.ctypeslib.ndpointer(ndim=-1)
print(f"Created: {ptr}")
print(f"ndim: {ptr._ndim_}")

arr = np.array([1, 2, 3])
try:
    ptr.from_param(arr)
except TypeError as e:
    print(f"Error: {e}")
```

Output:
```
Created: <class 'numpy.ctypeslib._ctypeslib.ndpointer_any_-1d'>
ndim: -1
Error: array must have -1 dimension(s)
```

The pointer is created successfully but produces a nonsensical error message when used.

## Why This Is A Bug

1. Negative dimensions are semantically invalid - arrays cannot have negative dimensionality
2. The function should validate parameters at creation time and raise ValueError/TypeError
3. The resulting error message "array must have -1 dimension(s)" is confusing to users
4. This violates the principle of early validation - invalid inputs should be rejected when the pointer type is created, not later during validation

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -289,6 +289,9 @@ def ndpointer(dtype=None, ndim=None, shape=None, flags=None):

     # normalize dtype to dtype | None
     if dtype is not None:
         dtype = np.dtype(dtype)
+
+    if ndim is not None and ndim < 0:
+        raise ValueError(f"ndim must be non-negative, got {ndim}")

     # normalize flags to int | None
```