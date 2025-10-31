# Bug Report: numpy.ctypeslib.as_ctypes Fortran-Ordered Arrays

**Target**: `numpy.ctypeslib.as_ctypes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`as_ctypes()` incorrectly rejects Fortran-ordered (column-major) arrays as "strided arrays not supported", even though F-contiguous arrays are just as valid and contiguous as C-contiguous arrays.

## Property-Based Test

```python
import numpy as np
import numpy.ctypeslib
from hypothesis import given, settings
import hypothesis.extra.numpy as npst


@settings(max_examples=300)
@given(npst.arrays(dtype=npst.scalar_dtypes(), shape=npst.array_shapes(min_dims=2)))
def test_as_ctypes_supports_fortran_order(arr):
    f_arr = np.asfortranarray(arr)
    ct = np.ctypeslib.as_ctypes(f_arr)
```

**Failing input**: `arr=array([[False, False], [False, False]])`

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib

arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
f_arr = np.asfortranarray(arr)

np.ctypeslib.as_ctypes(arr)

np.ctypeslib.as_ctypes(f_arr)
```

## Why This Is A Bug

1. Fortran-ordered arrays are contiguous in memory (column-major order) and should be just as valid as C-ordered arrays (row-major order).

2. The code checks `if ai["strides"]:` and rejects any array with non-None strides. However, C-contiguous arrays have `strides=None` in their `__array_interface__` (as an optimization), while F-contiguous arrays explicitly report their strides. This conflates "has explicit strides" with "is strided/non-contiguous".

3. Users working with Fortran libraries or numpy's Fortran-order arrays would reasonably expect to convert them to ctypes.

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -587,7 +587,8 @@ def as_ctypes(obj):

         """
         ai = obj.__array_interface__
-        if ai["strides"]:
+        if ai["strides"] and not (hasattr(obj, 'flags') and
+                                   (obj.flags.c_contiguous or obj.flags.f_contiguous)):
             raise TypeError("strided arrays not supported")
         if ai["version"] != 3:
             raise TypeError("only __array_interface__ version 3 supported")
```