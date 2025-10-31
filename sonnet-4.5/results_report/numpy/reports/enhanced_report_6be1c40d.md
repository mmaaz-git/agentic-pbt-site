# Bug Report: numpy.ctypeslib.as_array Adds Extra Dimensions When Converting Pointers to Multi-Dimensional Arrays

**Target**: `numpy.ctypeslib.as_array`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.ctypeslib.as_array()` function incorrectly adds extra dimensions when converting pointers to multi-dimensional ctypes arrays, violating its documented behavior of creating arrays with user-specified shapes.

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

if __name__ == "__main__":
    test_as_array_from_pointer_with_shape()
```

<details>

<summary>
**Failing input**: `arr=array([[False]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 25, in <module>
    test_as_array_from_pointer_with_shape()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 9, in test_as_array_from_pointer_with_shape
    def test_as_array_from_pointer_with_shape(arr):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 21, in test_as_array_from_pointer_with_shape
    np.testing.assert_array_equal(result, arr)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1051, in assert_array_equal
    assert_array_compare(operator.__eq__, actual, desired, err_msg=err_msg,
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         verbose=verbose, header='Arrays are not equal',
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         strict=strict)
                         ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 803, in assert_array_compare
    raise AssertionError(msg)
AssertionError:
Arrays are not equal

(shapes (1, 1, 1), (1, 1) mismatch)
 ACTUAL: array([[[False]]])
 DESIRED: array([[False]])
Falsifying example: test_as_array_from_pointer_with_shape(
    arr=array([[False]]),
)
```
</details>

## Reproducing the Bug

```python
import ctypes
import numpy as np

# Test case 1: 2D array with shape (1, 1)
print("Test case 1: 2D array with shape (1, 1)")
arr = np.array([[False]], dtype=np.bool_)
print(f"Original array: {arr}")
print(f"Original shape: {arr.shape}")

ct_arr = np.ctypeslib.as_ctypes(arr)
print(f"ctypes array type: {type(ct_arr)}")

ptr = ctypes.cast(ct_arr, ctypes.POINTER(ct_arr._type_))
print(f"Pointer type: {type(ptr)}")
print(f"Pointer element type: {ptr._type_}")

result = np.ctypeslib.as_array(ptr, shape=(1, 1))
print(f"Result array: {result}")
print(f"Result shape: {result.shape}")
print(f"Expected shape: (1, 1)")
print(f"Shape mismatch: {result.shape != (1, 1)}")
print()

# Test case 2: 2D array with shape (2, 2)
print("Test case 2: 2D array with shape (2, 2)")
arr2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
print(f"Original array:\n{arr2}")
print(f"Original shape: {arr2.shape}")

ct_arr2 = np.ctypeslib.as_ctypes(arr2)
ptr2 = ctypes.cast(ct_arr2, ctypes.POINTER(ct_arr2._type_))
result2 = np.ctypeslib.as_array(ptr2, shape=(2, 2))
print(f"Result array:\n{result2}")
print(f"Result shape: {result2.shape}")
print(f"Expected shape: (2, 2)")
print(f"Shape mismatch: {result2.shape != (2, 2)}")
print()

# Test case 3: 1D array (should work correctly)
print("Test case 3: 1D array with shape (3,)")
arr3 = np.array([1, 2, 3], dtype=np.int32)
print(f"Original array: {arr3}")
print(f"Original shape: {arr3.shape}")

ct_arr3 = np.ctypeslib.as_ctypes(arr3)
ptr3 = ctypes.cast(ct_arr3, ctypes.POINTER(ct_arr3._type_))
result3 = np.ctypeslib.as_array(ptr3, shape=(3,))
print(f"Result array: {result3}")
print(f"Result shape: {result3.shape}")
print(f"Expected shape: (3,)")
print(f"Shape mismatch: {result3.shape != (3,)}")
```

<details>

<summary>
Shape mismatch error - arrays get extra dimensions
</summary>
```
Test case 1: 2D array with shape (1, 1)
Original array: [[False]]
Original shape: (1, 1)
ctypes array type: <class 'c_bool_Array_1_Array_1'>
Pointer type: <class '__main__.LP_c_bool_Array_1'>
Pointer element type: <class 'c_bool_Array_1'>
Result array: [[[False]]]
Result shape: (1, 1, 1)
Expected shape: (1, 1)
Shape mismatch: True

Test case 2: 2D array with shape (2, 2)
Original array:
[[1 2]
 [3 4]]
Original shape: (2, 2)
Result array:
[[[     1      2]
  [     3      4]]

 [[233211      0]
  [    33      0]]]
Result shape: (2, 2, 2)
Expected shape: (2, 2)
Shape mismatch: True

Test case 3: 1D array with shape (3,)
Original array: [1 2 3]
Original shape: (3,)
Result array: [1 2 3]
Result shape: (3,)
Expected shape: (3,)
Shape mismatch: False
```
</details>

## Why This Is A Bug

The `as_array()` function is documented to "Create a numpy array from a ctypes array or POINTER" and explicitly states that "The shape parameter must be given if converting from a ctypes POINTER." The function's contract is to create an array with the exact shape specified by the user when converting from a pointer.

However, when given a pointer to a multi-dimensional ctypes array type (e.g., `POINTER(c_bool_Array_1)`), the function incorrectly treats the already-structured array type as if it were a scalar, then wraps it with the requested dimensions, resulting in extra dimensions. For example:
- A pointer to a 1D array type with requested shape `(1, 1)` produces shape `(1, 1, 1)` instead
- A pointer to a 2D array type's inner dimension with requested shape `(2, 2)` produces shape `(2, 2, 2)` instead

This violates the fundamental contract that the `shape` parameter controls the output array's shape. The documentation makes no distinction between pointers to scalar types versus pointers to array types - it simply promises to create arrays with the specified shape.

## Relevant Context

The bug occurs in the `as_array()` implementation at `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py:556`:

```python
if isinstance(obj, ctypes._Pointer):
    if shape is None:
        raise TypeError('as_array() requires a shape argument when called on a pointer')
    p_arr_type = ctypes.POINTER(_ctype_ndarray(obj._type_, shape))
    obj = ctypes.cast(obj, p_arr_type).contents
```

The issue is that when `obj._type_` is already an array type (like `c_bool_Array_1`), the `_ctype_ndarray()` function (lines 357-363) incorrectly nests it further:

```python
def _ctype_ndarray(element_type, shape):
    """ Create an ndarray of the given element type and shape """
    for dim in shape[::-1]:
        element_type = dim * element_type
        element_type.__module__ = None
    return element_type
```

This function assumes `element_type` is always a scalar type, but when it's already an array type, it creates nested arrays with too many dimensions.

**Documentation reference**: The function documentation (lines 521-549) provides examples only with pointers to scalar types, never with pointers to array types, leaving this behavior unspecified but still incorrect given the stated contract.

**Workaround**: Users can avoid this bug by using `np.asarray(ct_arr)` directly on the ctypes array without the pointer conversion step.

## Proposed Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -549,10 +549,20 @@ if ctypes is not None:
         """
         if isinstance(obj, ctypes._Pointer):
             # convert pointers to an array of the desired shape
             if shape is None:
                 raise TypeError(
                     'as_array() requires a shape argument when called on a '
                     'pointer')
+
+            # Extract the base scalar type if obj._type_ is an array
+            element_type = obj._type_
+            while hasattr(element_type, '_type_'):
+                element_type = element_type._type_
+
-            p_arr_type = ctypes.POINTER(_ctype_ndarray(obj._type_, shape))
+            p_arr_type = ctypes.POINTER(_ctype_ndarray(element_type, shape))
             obj = ctypes.cast(obj, p_arr_type).contents

         return np.asarray(obj)
```