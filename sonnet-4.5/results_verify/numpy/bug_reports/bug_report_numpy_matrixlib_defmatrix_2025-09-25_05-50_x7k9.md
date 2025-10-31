# Bug Report: numpy.matrixlib Dead Code in matrix.__new__

**Target**: `numpy.matrixlib.defmatrix.matrix.__new__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Line 166 in `defmatrix.py` contains a logic error that makes the copy operation on line 167 unreachable dead code.

## Property-Based Test

```python
import numpy as np
from hypothesis import given
from hypothesis.extra import numpy as hnp


@given(hnp.arrays(dtype=np.float64, shape=(4, 4)))
def test_dead_code_bug_noncontiguous_copy(arr):
    non_contiguous = arr[::2, ::2]

    if not non_contiguous.flags.contiguous:
        m = np.matrix(non_contiguous, copy=False)

        if not m.flags.c_contiguous and not m.flags.f_contiguous:
            assert False, "Matrix from non-contiguous array is non-contiguous"
```

**Failing input**: Any non-contiguous array, e.g., `np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])[::2, ::2]`

## Reproducing the Bug

```python
import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
non_contiguous = arr[::2, ::2]

print(f"Input contiguous: {non_contiguous.flags.contiguous}")

m = np.matrix(non_contiguous, copy=False)

print(f"Matrix contiguous: {m.flags.contiguous}")
```

## Why This Is A Bug

At line 166, the condition `if not (order or arr.flags.contiguous):` uses boolean OR with `order`, which is always 'C' or 'F' (truthy strings). This makes the condition always evaluate to False, rendering line 167 (`arr = arr.copy()`) unreachable.

The variable `order` is assigned at line 162-164:
```python
order = 'C'
if (ndim == 2) and arr.flags.fortran:
    order = 'F'
```

Since `order` is always a non-empty string (truthy), the expression `(order or arr.flags.contiguous)` is always truthy, making `not (order or arr.flags.contiguous)` always False.

## Fix

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -163,7 +163,7 @@ class matrix(N.ndarray):
         if (ndim == 2) and arr.flags.fortran:
             order = 'F'

-        if not (order or arr.flags.contiguous):
+        if not arr.flags.contiguous:
             arr = arr.copy()

         ret = N.ndarray.__new__(subtype, shape, arr.dtype,
```