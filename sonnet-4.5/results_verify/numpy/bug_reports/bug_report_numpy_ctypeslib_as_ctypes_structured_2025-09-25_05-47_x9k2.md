# Bug Report: numpy.ctypeslib.as_ctypes Structured Array Support

**Target**: `numpy.ctypeslib.as_ctypes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.ctypeslib.as_ctypes()` fails to convert structured numpy arrays to ctypes objects, despite `numpy.ctypeslib.as_ctypes_type()` supporting structured dtypes. This inconsistency breaks the expected API contract and prevents valid use cases.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
import numpy.ctypeslib


@settings(max_examples=100)
@given(
    x=st.integers(min_value=-1000, max_value=1000),
    y=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
)
def test_as_ctypes_structured_array_unsupported(x, y):
    dt = np.dtype([('x', 'i4'), ('y', 'f4')])
    arr = np.array([(x, y)], dtype=dt)

    numpy.ctypeslib.as_ctypes_type(dt)

    try:
        numpy.ctypeslib.as_ctypes(arr)
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        pass
```

**Failing input**: `x=0, y=0.0` (or any values)

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib

dt = np.dtype([('x', 'i4'), ('y', 'f4')])
arr = np.array([(1, 2.0), (3, 4.0)], dtype=dt)

print('as_ctypes_type works:')
ctype = numpy.ctypeslib.as_ctypes_type(dt)
print(f'  {ctype}')

print('\nas_ctypes fails:')
try:
    ctypes_obj = numpy.ctypeslib.as_ctypes(arr)
except NotImplementedError as e:
    print(f'  NotImplementedError: {e}')
```

## Why This Is A Bug

The `as_ctypes` function uses `ai["typestr"]` from the `__array_interface__` protocol, which for structured arrays is a void type (e.g., `'|V8'`) that loses all field information. However:

1. The `__array_interface__` provides full dtype information in `ai["descr"]`
2. The `as_ctypes_type()` function successfully converts structured dtypes to ctypes structures
3. This creates an API inconsistency where `as_ctypes_type(dtype)` works but `as_ctypes(array_with_that_dtype)` doesn't

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -594,9 +594,14 @@ def as_ctypes(obj):
     if ai["version"] != 3:
         raise TypeError("only __array_interface__ version 3 supported")
     addr, readonly = ai["data"]
     if readonly:
         raise TypeError("readonly arrays unsupported")

-    # can't use `_dtype((ai["typestr"], ai["shape"]))` here, as it overflows
-    # dtype.itemsize (gh-14214)
-    ctype_scalar = as_ctypes_type(ai["typestr"])
+    # Use the full dtype instead of just typestr to preserve structured array info
+    # Note: We get the dtype from the original object rather than reconstructing
+    # from __array_interface__ to preserve all field information
+    if hasattr(obj, 'dtype'):
+        ctype_scalar = as_ctypes_type(obj.dtype)
+    else:
+        # Fallback to typestr for objects without dtype attribute
+        ctype_scalar = as_ctypes_type(ai["typestr"])
     result_type = _ctype_ndarray(ctype_scalar, ai["shape"])
     result = result_type.from_address(addr)
     result.__keep = obj
```