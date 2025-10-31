# Bug Report: numpy.ctypeslib.as_array Incorrect Shape When Converting Pointers to Multi-Dimensional Arrays

**Target**: `numpy.ctypeslib.as_array`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.ctypeslib.as_array()` produces arrays with incorrect shapes when called with pointers to multi-dimensional ctypes arrays. For example, converting a pointer to a `(1, 1)` array produces a `(1, 1, 1)` array instead.

## Property-Based Test

```python
import ctypes

import numpy as np
from hypothesis import assume, given
from hypothesis.extra import numpy as npst


@given(npst.arrays(dtype=npst.scalar_dtypes(), shape=npst.array_shapes(min_dims=1, max_dims=3)))
def test_as_array_from_pointer_with_shape(arr):
    assume(arr.flags.c_contiguous)
    assume(arr.dtype.hasobject == False)

    try:
        ct_arr = np.ctypeslib.as_ctypes(arr)
    except (TypeError, NotImplementedError):
        assume(False)

    ptr = ctypes.cast(ct_arr, ctypes.POINTER(ct_arr._type_))
    result = np.ctypeslib.as_array(ptr, arr.shape)

    np.testing.assert_array_equal(result, arr)
    assert result.shape == arr.shape
```

**Failing input**: `arr=array([[False]])`

## Reproducing the Bug

```python
import ctypes

import numpy as np

arr = np.array([[False]], dtype=np.bool_)
ct_arr = np.ctypeslib.as_ctypes(arr)
ptr = ctypes.cast(ct_arr, ctypes.POINTER(ct_arr._type_))

result = np.ctypeslib.as_array(ptr, shape=(1, 1))
print(f"Expected shape: (1, 1)")
print(f"Actual shape: {result.shape}")
```

## Why This Is A Bug

The `as_array()` function is documented to convert ctypes pointers to numpy arrays with a specified shape. When given a pointer to a multi-dimensional ctypes array and a shape argument, it should produce an array with that exact shape. Instead, it adds extra dimensions because it treats the pointer's element type (which is already an array) as a scalar, wrapping it with additional dimensions.

The root cause is in the `as_array()` implementation at line 556:
```python
p_arr_type = ctypes.POINTER(_ctype_ndarray(obj._type_, shape))
```

When `obj._type_` is already an array type (e.g., `c_bool_Array_1`), `_ctype_ndarray` wraps it in additional dimensions instead of recognizing it as already having dimensions.

## Fix

The fix should detect when `obj._type_` is already an array type and extract the base scalar type before calling `_ctype_ndarray`:

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -549,10 +549,18 @@ if ctypes is not None:
         """
         if isinstance(obj, ctypes._Pointer):
             # convert pointers to an array of the desired shape
             if shape is None:
                 raise TypeError(
                     'as_array() requires a shape argument when called on a '
                     'pointer')
+
+            # Extract the base element type if obj._type_ is an array
+            element_type = obj._type_
+            while hasattr(element_type, '_type_'):
+                element_type = element_type._type_
+
             p_arr_type = ctypes.POINTER(_ctype_ndarray(obj._type_, shape))
             obj = ctypes.cast(obj, p_arr_type).contents

         return np.asarray(obj)
```

Note: The actual fix would need `_ctype_ndarray(element_type, shape)` instead of `_ctype_ndarray(obj._type_, shape)`.