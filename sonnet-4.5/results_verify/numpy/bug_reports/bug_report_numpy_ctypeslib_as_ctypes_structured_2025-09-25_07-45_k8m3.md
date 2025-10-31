# Bug Report: numpy.ctypeslib.as_ctypes Structured Dtype

**Target**: `numpy.ctypeslib.as_ctypes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`as_ctypes()` fails with NotImplementedError on structured dtypes, even though the companion function `as_ctypes_type()` successfully handles them and the docstring claims to accept "anything that exposes the __array_interface__".

## Property-Based Test

```python
import numpy as np
import numpy.ctypeslib
from hypothesis import given, strategies as st, settings


supported_scalars = [
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float32, np.float64,
    np.bool_,
]


@st.composite
def structured_dtypes(draw):
    num_fields = draw(st.integers(min_value=1, max_value=5))
    fields = []
    for i in range(num_fields):
        field_name = f"f{i}"
        field_dtype = draw(st.sampled_from(supported_scalars))
        fields.append((field_name, field_dtype))
    return np.dtype(fields)


@settings(max_examples=300)
@given(structured_dtypes(), st.integers(min_value=1, max_value=10))
def test_as_ctypes_supports_structured_dtypes(dtype, size):
    arr = np.zeros(size, dtype=dtype)
    ct = np.ctypeslib.as_ctypes(arr)
```

**Failing input**: `dtype=[('f0', 'i1')], size=1`

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib

dtype = np.dtype([('x', np.int32), ('y', np.float64)])
arr = np.zeros(3, dtype=dtype)

np.ctypeslib.as_ctypes_type(arr.dtype)

np.ctypeslib.as_ctypes(arr)
```

## Why This Is A Bug

1. The `as_ctypes()` docstring states it accepts "anything that exposes the __array_interface__", and structured arrays expose this interface.

2. The companion function `as_ctypes_type()` successfully converts structured dtypes to ctypes types via `_ctype_from_dtype_structured()`, demonstrating that the module is designed to support structured dtypes.

3. Users would reasonably expect to convert structured numpy arrays to ctypes for interfacing with C libraries.

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -596,7 +596,12 @@ def as_ctypes(obj):
         raise TypeError("readonly arrays unsupported")

-        ctype_scalar = as_ctypes_type(ai["typestr"])
+        if hasattr(obj, 'dtype'):
+            ctype_scalar = as_ctypes_type(obj.dtype)
+        elif "descr" in ai and ai["descr"]:
+            ctype_scalar = as_ctypes_type(ai["descr"])
+        else:
+            ctype_scalar = as_ctypes_type(ai["typestr"])
         result_type = _ctype_ndarray(ctype_scalar, ai["shape"])
         result = result_type.from_address(addr)
         result.__keep = obj
```