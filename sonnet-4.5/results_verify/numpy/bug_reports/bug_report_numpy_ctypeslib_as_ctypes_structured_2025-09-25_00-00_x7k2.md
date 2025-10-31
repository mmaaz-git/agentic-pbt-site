# Bug Report: numpy.ctypeslib.as_ctypes Fails on Structured Arrays

**Target**: `numpy.ctypeslib.as_ctypes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.ctypeslib.as_ctypes()` fails with `NotImplementedError` when given structured arrays, even though `as_ctypes_type()` successfully handles structured dtypes. This is an inconsistency that breaks the expected workflow for converting structured NumPy arrays to ctypes for C library interop.

## Property-Based Test

```python
import numpy as np
import numpy.ctypeslib as npc
from hypothesis import given, assume
import hypothesis.extra.numpy as npst

@given(npst.arrays(dtype=st.just(np.dtype([('x', np.int32), ('y', np.float64)])),
                   shape=npst.array_shapes(min_dims=1, max_dims=2, max_side=10)))
def test_as_ctypes_structured_array(arr):
    assume(arr.flags.c_contiguous)
    assume(arr.flags.writeable)

    c_arr = npc.as_ctypes(arr)
    recovered = npc.as_array(c_arr)

    assert np.array_equal(recovered, arr)
```

**Failing input**: `np.array([(1, 2.0)], dtype=[('x', np.int32), ('y', np.float64)])`

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib as npc

arr = np.array([(1, 2.0)], dtype=[('x', np.int32), ('y', np.float64)])

ctype = npc.as_ctypes_type(arr.dtype)
print(f"as_ctypes_type works: {ctype}")

c_arr = npc.as_ctypes(arr)
```

Output:
```
as_ctypes_type works: <class 'struct'>
Traceback (most recent call last):
  File "repro.py", line 8, in <module>
    c_arr = npc.as_ctypes(arr)
  File ".../numpy/ctypeslib/_ctypeslib.py", line 599, in as_ctypes
    ctype_scalar = as_ctypes_type(ai["typestr"])
NotImplementedError: Converting dtype('V12') to a ctypes type
```

## Why This Is A Bug

1. **Inconsistent API**: `as_ctypes_type()` successfully converts structured dtypes to ctypes structures, but `as_ctypes()` fails on structured arrays
2. **Breaks documented workflow**: The module documentation shows using `as_ctypes_type` to define types and `as_ctypes` to convert arrays for C interop
3. **No documented limitation**: The `as_ctypes()` docstring doesn't mention that structured arrays are unsupported
4. **Real-world impact**: Users working with C libraries that expect struct arrays cannot use the natural NumPy→ctypes conversion

The root cause is in line 599 of `_ctypeslib.py`: the code uses `ai["typestr"]` which for structured arrays is a void type like `|V12`, rather than the actual structured dtype.

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -596,7 +596,10 @@ def as_ctypes(obj):

         # can't use `_dtype((ai["typestr"], ai["shape"]))` here, as it overflows
         # dtype.itemsize (gh-14214)
-        ctype_scalar = as_ctypes_type(ai["typestr"])
+        if hasattr(obj, 'dtype'):
+            ctype_scalar = as_ctypes_type(obj.dtype)
+        else:
+            ctype_scalar = as_ctypes_type(ai["typestr"])
         result_type = _ctype_ndarray(ctype_scalar, ai["shape"])
         result = result_type.from_address(addr)
         result.__keep = obj
```

This fix uses the actual dtype when available (for NumPy arrays), which properly handles structured types, while falling back to typestr for generic objects that only expose `__array_interface__`.