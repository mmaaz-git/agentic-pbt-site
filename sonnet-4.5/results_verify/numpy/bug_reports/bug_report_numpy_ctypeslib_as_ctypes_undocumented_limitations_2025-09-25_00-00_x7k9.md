# Bug Report: numpy.ctypeslib.as_ctypes Undocumented Limitations

**Target**: `numpy.ctypeslib.as_ctypes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `numpy.ctypeslib.as_ctypes` function's docstring claims "anything that exposes the __array_interface__ is accepted", but the function actually rejects strided (non-contiguous) and readonly arrays without documenting these limitations.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings


@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=2, max_size=100))
@settings(max_examples=100)
def test_as_ctypes_accepts_strided_arrays(data):
    np_array = np.array(data, dtype=np.int32)
    sliced = np_array[::2]

    try:
        ctypes_array = np.ctypeslib.as_ctypes(sliced)
        assert True, "Strided array was accepted"
    except TypeError as e:
        assert False, f"as_ctypes docstring says 'anything that exposes __array_interface__ is accepted', but strided arrays are rejected: {e}"


@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100))
@settings(max_examples=100)
def test_as_ctypes_accepts_readonly_arrays(data):
    np_array = np.array(data, dtype=np.int32)
    np_array.flags.writeable = False

    try:
        ctypes_array = np.ctypeslib.as_ctypes(np_array)
        assert True, "Readonly array was accepted"
    except TypeError as e:
        assert False, f"as_ctypes docstring says 'anything that exposes __array_interface__ is accepted', but readonly arrays are rejected: {e}"
```

**Failing input**: Any strided array (e.g., `np.array([0, 0, 0])[::2]`) or readonly array

## Reproducing the Bug

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])
sliced = arr[::2]

np.ctypeslib.as_ctypes(sliced)
```

This raises `TypeError: strided arrays not supported`, despite the docstring claiming "anything that exposes the __array_interface__ is accepted".

Similarly:

```python
import numpy as np

arr = np.array([1, 2, 3])
arr.flags.writeable = False

np.ctypeslib.as_ctypes(arr)
```

This raises `TypeError: readonly arrays unsupported`.

## Why This Is A Bug

The docstring states: "Create and return a ctypes object from a numpy array. Actually anything that exposes the __array_interface__ is accepted."

However, the implementation explicitly rejects:
1. Strided arrays (non-contiguous arrays, including sliced arrays, transposed arrays, and Fortran-order arrays)
2. Readonly arrays

These are common numpy array types that users would reasonably expect to work based on the docstring. The limitations should be clearly documented in the function's docstring, including:
- A "Raises" section documenting the TypeError exceptions
- Clarification in the description about which arrays are supported

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -19,7 +19,13 @@ def as_ctypes(obj):
     """
     Create and return a ctypes object from a numpy array.  Actually
-    anything that exposes the __array_interface__ is accepted.
+    anything that exposes the __array_interface__ is accepted, with
+    some limitations: only C-contiguous, writeable arrays are supported.
+
+    .. note::
+       This function does not support strided (non-contiguous) arrays,
+       such as sliced arrays, transposed arrays, or Fortran-order arrays.
+       It also does not support readonly arrays.

     Examples
     --------
@@ -38,6 +44,16 @@ def as_ctypes(obj):
     >>> c_int_array[:]
     [1, 2, 3]

+    Raises
+    ------
+    TypeError
+        If the array is strided (non-contiguous), such as a sliced,
+        transposed, or Fortran-order array.
+    TypeError
+        If the array is readonly (writeable flag is False).
+    TypeError
+        If the array interface version is not 3.
+
     """
     ai = obj.__array_interface__
     if ai["strides"]: