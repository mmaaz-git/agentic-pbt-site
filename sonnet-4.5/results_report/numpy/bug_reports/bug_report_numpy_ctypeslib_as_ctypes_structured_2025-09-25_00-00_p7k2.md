# Bug Report: numpy.ctypeslib.as_ctypes Structured Array Support

**Target**: `numpy.ctypeslib.as_ctypes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.ctypeslib.as_ctypes` fails to convert numpy arrays with structured dtypes to ctypes objects, even though the underlying `as_ctypes_type` function fully supports structured dtypes. The bug occurs because `as_ctypes` passes the typestr from `__array_interface__` (e.g., '|V12') instead of the actual dtype object.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings


@given(
    st.lists(
        st.tuples(
            st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122),
                    min_size=1, max_size=10),
            st.sampled_from([np.int32, np.float64])
        ),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0]
    )
)
@settings(max_examples=200)
def test_structured_array_conversion(field_specs):
    dtype = np.dtype([(name, dt) for name, dt in field_specs])
    arr = np.zeros(10, dtype=dtype)

    ctypes_obj = np.ctypeslib.as_ctypes(arr)
    result = np.ctypeslib.as_array(ctypes_obj)

    for name, _ in field_specs:
        np.testing.assert_array_equal(result[name], arr[name])
```

**Failing input**: Any structured dtype, e.g., `field_specs=[('a', numpy.int32)]`

## Reproducing the Bug

```python
import numpy as np

dtype = np.dtype([('x', np.int32), ('y', np.float64)])
arr = np.array([(1, 1.5), (2, 2.5)], dtype=dtype)

print(f"Array: {arr}")
print(f"Dtype: {arr.dtype}")

print("\nAttempting np.ctypeslib.as_ctypes(arr):")
try:
    ctypes_obj = np.ctypeslib.as_ctypes(arr)
    print(f"Success: {ctypes_obj}")
except NotImplementedError as e:
    print(f"NotImplementedError: {e}")
    print("\nHowever, as_ctypes_type CAN handle this dtype:")
    ctype = np.ctypeslib.as_ctypes_type(arr.dtype)
    print(f"  as_ctypes_type(arr.dtype) = {ctype}")
```

Output:
```
Array: [(1, 1.5) (2, 2.5)]
Dtype: [('x', '<i4'), ('y', '<f8')]

Attempting np.ctypeslib.as_ctypes(arr):
NotImplementedError: Converting dtype('V12') to a ctypes type

However, as_ctypes_type CAN handle this dtype:
  as_ctypes_type(arr.dtype) = <class 'struct'>
```

## Why This Is A Bug

1. **Inconsistent API**: `as_ctypes_type` successfully handles structured dtypes, so users would reasonably expect `as_ctypes` to work as well.

2. **Implementation error**: The bug is in line 599 of `_ctypeslib.py` where `as_ctypes` calls `as_ctypes_type(ai["typestr"])`. For structured arrays, the typestr is a void type (e.g., '|V12') that doesn't contain field information, causing the conversion to fail. The function should use `obj.dtype` instead.

3. **Common use case**: Structured arrays are a standard NumPy feature for representing C structs, and the inability to convert them to ctypes undermines the module's purpose of interfacing with C libraries.

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -596,7 +596,7 @@ def as_ctypes(obj):

         # can't use `_dtype((ai["typestr"], ai["shape"]))` here, as it overflows
         # dtype.itemsize (gh-14214)
-        ctype_scalar = as_ctypes_type(ai["typestr"])
+        ctype_scalar = as_ctypes_type(obj.dtype)
         result_type = _ctype_ndarray(ctype_scalar, ai["shape"])
         result = result_type.from_address(addr)
         result.__keep = obj
```

This fix passes the actual dtype object instead of the typestr, allowing `as_ctypes_type` to properly handle structured dtypes using the full field information.