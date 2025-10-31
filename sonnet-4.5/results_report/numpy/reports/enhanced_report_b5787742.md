# Bug Report: numpy.ctypeslib.as_ctypes Rejects Fortran-Ordered Arrays

**Target**: `numpy.ctypeslib.as_ctypes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`as_ctypes()` incorrectly rejects Fortran-ordered (column-major) arrays as "strided arrays not supported", even though F-contiguous arrays are contiguous in memory just like C-contiguous arrays.

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


if __name__ == "__main__":
    test_as_ctypes_supports_fortran_order()
```

<details>

<summary>
**Failing input**: `arr=array([[False, False], [False, False]])`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 15, in <module>
  |     test_as_ctypes_supports_fortran_order()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 8, in test_as_ctypes_supports_fortran_order
  |     @given(npst.arrays(dtype=npst.scalar_dtypes(), shape=npst.array_shapes(min_dims=2)))
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 11, in test_as_ctypes_supports_fortran_order
    |     ct = np.ctypeslib.as_ctypes(f_arr)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 599, in as_ctypes
    |     ctype_scalar = as_ctypes_type(ai["typestr"])
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 518, in as_ctypes_type
    |     return _ctype_from_dtype(np.dtype(dtype))
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 461, in _ctype_from_dtype
    |     return _ctype_from_dtype_scalar(dtype)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 387, in _ctype_from_dtype_scalar
    |     raise NotImplementedError(
    |         f"Converting {dtype!r} to a ctypes type"
    |     ) from None
    | NotImplementedError: Converting dtype('float16') to a ctypes type
    | Falsifying example: test_as_ctypes_supports_fortran_order(
    |     arr=array([[0.]], dtype=float16),
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 11, in test_as_ctypes_supports_fortran_order
    |     ct = np.ctypeslib.as_ctypes(f_arr)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py", line 590, in as_ctypes
    |     raise TypeError("strided arrays not supported")
    | TypeError: strided arrays not supported
    | Falsifying example: test_as_ctypes_supports_fortran_order(
    |     arr=array([[False, False],
    |            [False, False]]),
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib

# Create a simple 2x2 integer array
arr = np.array([[1, 2], [3, 4]], dtype=np.int32)

# Convert to Fortran-ordered (column-major) array
f_arr = np.asfortranarray(arr)

# Check array flags
print("Original array (C-contiguous):")
print(f"  Flags: C_CONTIGUOUS={arr.flags.c_contiguous}, F_CONTIGUOUS={arr.flags.f_contiguous}")
print(f"  __array_interface__['strides'] = {arr.__array_interface__['strides']}")

print("\nFortran-ordered array:")
print(f"  Flags: C_CONTIGUOUS={f_arr.flags.c_contiguous}, F_CONTIGUOUS={f_arr.flags.f_contiguous}")
print(f"  __array_interface__['strides'] = {f_arr.__array_interface__['strides']}")

# Try to convert C-contiguous array to ctypes (should work)
print("\nConverting C-contiguous array to ctypes:")
try:
    c_ctypes = np.ctypeslib.as_ctypes(arr)
    print(f"  Success! Type: {type(c_ctypes)}")
except TypeError as e:
    print(f"  Failed: {e}")

# Try to convert F-contiguous array to ctypes (will fail with current implementation)
print("\nConverting F-contiguous array to ctypes:")
try:
    f_ctypes = np.ctypeslib.as_ctypes(f_arr)
    print(f"  Success! Type: {type(f_ctypes)}")
except TypeError as e:
    print(f"  Failed: {e}")
```

<details>

<summary>
TypeError when converting F-contiguous array
</summary>
```
Original array (C-contiguous):
  Flags: C_CONTIGUOUS=True, F_CONTIGUOUS=False
  __array_interface__['strides'] = None

Fortran-ordered array:
  Flags: C_CONTIGUOUS=False, F_CONTIGUOUS=True
  __array_interface__['strides'] = (4, 8)

Converting C-contiguous array to ctypes:
  Success! Type: <class 'c_int_Array_2_Array_2'>

Converting F-contiguous array to ctypes:
  Failed: strided arrays not supported
```
</details>

## Why This Is A Bug

1. **Both C-ordered and F-ordered arrays are contiguous in memory**. According to NumPy's documentation, "Both the C and Fortran orders are contiguous, i.e., single-segment, memory layouts." F-contiguous arrays store data column-by-column instead of row-by-row, but the memory is still contiguous.

2. **The implementation incorrectly conflates "has explicit strides" with "is non-contiguous"**. The code at line 589 checks `if ai["strides"]:` and rejects any array with non-None strides. However:
   - C-contiguous arrays report `strides=None` in their `__array_interface__` as an optimization
   - F-contiguous arrays explicitly report their strides (e.g., `(4, 8)` for a 2x2 int32 array)
   - Having explicit strides doesn't mean the array is non-contiguous

3. **The function documentation doesn't restrict to C-contiguous arrays**. The docstring states it accepts "anything that exposes the __array_interface__", and F-contiguous arrays do expose this interface.

4. **The error message is misleading**. The error "strided arrays not supported" implies non-contiguous arrays are rejected, but F-contiguous arrays ARE contiguous, just with a different memory layout.

## Relevant Context

- NumPy explicitly provides functions like `np.asfortranarray()` to create Fortran-ordered arrays
- Scientific computing often involves interfacing with Fortran libraries that expect column-major ordering
- The `ndpointer` function in the same module supports both C_CONTIGUOUS and F_CONTIGUOUS flags, showing that ctypeslib is intended to work with both orderings
- Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py:589-590`
- NumPy array documentation: https://numpy.org/doc/stable/reference/arrays.ndarray.html#array-attributes

Users can work around this bug by converting to C-contiguous first: `np.ctypeslib.as_ctypes(np.ascontiguousarray(f_arr))`, but this creates an unnecessary copy.

## Proposed Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -586,8 +586,11 @@ def as_ctypes(obj):

         """
         ai = obj.__array_interface__
-        if ai["strides"]:
-            raise TypeError("strided arrays not supported")
+        # Check if array is truly non-contiguous (not just F-contiguous)
+        if ai["strides"] and hasattr(obj, 'flags'):
+            if not (obj.flags.c_contiguous or obj.flags.f_contiguous):
+                raise TypeError("strided arrays not supported")
+        # For objects without flags, accept if strides is None (backwards compatibility)
         if ai["version"] != 3:
             raise TypeError("only __array_interface__ version 3 supported")
         addr, readonly = ai["data"]
```