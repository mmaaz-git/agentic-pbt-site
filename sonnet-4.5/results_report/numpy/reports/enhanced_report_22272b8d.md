# Bug Report: numpy.ctypeslib.as_ctypes Fails on Structured Arrays

**Target**: `numpy.ctypeslib.as_ctypes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.ctypeslib.as_ctypes()` fails with `NotImplementedError` when converting structured NumPy arrays to ctypes, despite `as_ctypes_type()` successfully handling the same structured dtypes.

## Property-Based Test

```python
import numpy as np
import numpy.ctypeslib as npc
from hypothesis import given, assume, strategies as st
import hypothesis.extra.numpy as npst

@given(npst.arrays(dtype=st.just(np.dtype([('x', np.int32), ('y', np.float64)])),
                   shape=npst.array_shapes(min_dims=1, max_dims=2, max_side=10)))
def test_as_ctypes_structured_array(arr):
    """Test that as_ctypes works with structured arrays when flags allow it."""
    # Only test arrays that meet the requirements for as_ctypes
    assume(arr.flags.c_contiguous)
    assume(arr.flags.writeable)

    # This should work since as_ctypes_type works with structured dtypes
    c_arr = npc.as_ctypes(arr)

    # Verify round-trip conversion
    recovered = npc.as_array(c_arr)
    assert np.array_equal(recovered, arr)

if __name__ == "__main__":
    test_as_ctypes_structured_array()
```

<details>

<summary>
**Failing input**: `array([(0, 0.)], dtype=[('x', '<i4'), ('y', '<f8')])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 22, in <module>
    test_as_ctypes_structured_array()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 7, in test_as_ctypes_structured_array
    shape=npst.array_shapes(min_dims=1, max_dims=2, max_side=10)))

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 15, in test_as_ctypes_structured_array
    c_arr = npc.as_ctypes(arr)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 599, in as_ctypes
    ctype_scalar = as_ctypes_type(ai["typestr"])
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 518, in as_ctypes_type
    return _ctype_from_dtype(np.dtype(dtype))
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 461, in _ctype_from_dtype
    return _ctype_from_dtype_scalar(dtype)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 387, in _ctype_from_dtype_scalar
    raise NotImplementedError(
        f"Converting {dtype!r} to a ctypes type"
    ) from None
NotImplementedError: Converting dtype('V12') to a ctypes type
Falsifying example: test_as_ctypes_structured_array(
    arr=array([(0, 0.)], dtype=[('x', '<i4'), ('y', '<f8')]),  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib as npc

# Create a structured array
arr = np.array([(1, 2.0)], dtype=[('x', np.int32), ('y', np.float64)])
print(f"Original array: {arr}")
print(f"Array dtype: {arr.dtype}")
print(f"Array flags: C_CONTIGUOUS={arr.flags.c_contiguous}, WRITEABLE={arr.flags.writeable}")
print()

# Show that as_ctypes_type works on the structured dtype
try:
    ctype = npc.as_ctypes_type(arr.dtype)
    print(f"as_ctypes_type(arr.dtype) works: {ctype}")
    print(f"Created ctypes structure fields: {[field[0] for field in ctype._fields_]}")
    print()
except Exception as e:
    print(f"as_ctypes_type failed: {e}")
    print()

# Now try as_ctypes on the structured array - this will fail
try:
    print("Attempting npc.as_ctypes(arr)...")
    c_arr = npc.as_ctypes(arr)
    print(f"Success: {c_arr}")
except NotImplementedError as e:
    print(f"NotImplementedError raised: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
```

<details>

<summary>
NotImplementedError: Converting dtype('V12') to a ctypes type
</summary>
```
Original array: [(1, 2.)]
Array dtype: [('x', '<i4'), ('y', '<f8')]
Array flags: C_CONTIGUOUS=True, WRITEABLE=True

as_ctypes_type(arr.dtype) works: <class 'struct'>
Created ctypes structure fields: ['x', 'y']

Attempting npc.as_ctypes(arr)...
NotImplementedError raised: Converting dtype('V12') to a ctypes type
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **API Inconsistency**: The `as_ctypes_type()` function explicitly supports structured dtypes (documented with examples showing conversion of `dtype([('x', 'i4'), ('y', 'f4')])` to ctypes Structure), yet `as_ctypes()` fails on arrays with these same dtypes.

2. **Documentation Contradiction**: The `as_ctypes()` docstring states it accepts "anything that exposes the __array_interface__" without mentioning any limitation for structured arrays. Structured arrays do expose `__array_interface__`.

3. **Broken Workflow**: The natural workflow for C library interop would be:
   - Define structured dtype matching C struct layout
   - Create NumPy array with that dtype
   - Convert to ctypes for passing to C library
   - This workflow fails at the last step despite the infrastructure existing

4. **Implementation Bug**: The failure occurs at line 599 of `_ctypeslib.py` where the code uses `ai["typestr"]` from `__array_interface__`. For structured arrays, this returns a void type string like `"|V12"` (12-byte void type) rather than the actual structured dtype information. Meanwhile, the actual dtype object contains all necessary information and `as_ctypes_type()` can successfully convert it.

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py:599`. The `as_ctypes()` function gets the array's `__array_interface__` dictionary and attempts to convert `ai["typestr"]` to a ctypes type. However, for structured arrays, `typestr` is a void type (`"|V12"` for a 12-byte struct) that lacks the field information.

The sister function `as_ctypes_type()` works correctly because it receives the full dtype object with field information. The examples in its documentation specifically show converting structured dtypes to ctypes structures.

This is a common use case for developers working with C libraries that expect arrays of structs. The workaround requires manually creating ctypes structures and copying data, which defeats the purpose of the numpy.ctypeslib module's interoperability features.

Documentation reference: https://numpy.org/doc/stable/reference/routines.ctypes.html

## Proposed Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -596,7 +596,10 @@ def as_ctypes(obj):

         # can't use `_dtype((ai["typestr"], ai["shape"]))` here, as it overflows
         # dtype.itemsize (gh-14214)
-        ctype_scalar = as_ctypes_type(ai["typestr"])
+        if hasattr(obj, 'dtype') and obj.dtype.fields is not None:
+            ctype_scalar = as_ctypes_type(obj.dtype)
+        else:
+            ctype_scalar = as_ctypes_type(ai["typestr"])
         result_type = _ctype_ndarray(ctype_scalar, ai["shape"])
         result = result_type.from_address(addr)
         result.__keep = obj
```