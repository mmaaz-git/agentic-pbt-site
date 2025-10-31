# Bug Report: numpy.ctypeslib.ndpointer Accepts Invalid Negative Shape Dimensions

**Target**: `numpy.ctypeslib.ndpointer`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ndpointer` function incorrectly accepts negative values in the `shape` parameter without validation, creating pointer types with impossible-to-satisfy constraints that produce misleading error messages when used.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy.ctypeslib
import numpy as np


@given(st.tuples(st.integers(min_value=-10, max_value=-1), st.integers(min_value=1, max_value=10)))
@settings(max_examples=200)
def test_ndpointer_negative_shape(shape):
    try:
        ptr = numpy.ctypeslib.ndpointer(shape=shape)
        assert False, f"Should reject shape with negative dimensions {shape}"
    except (TypeError, ValueError):
        pass

if __name__ == "__main__":
    test_ndpointer_negative_shape()
```

<details>

<summary>
**Failing input**: `shape=(-1, 1)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 16, in <module>
    test_ndpointer_negative_shape()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 7, in test_ndpointer_negative_shape
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 11, in test_ndpointer_negative_shape
    assert False, f"Should reject shape with negative dimensions {shape}"
           ^^^^^
AssertionError: Should reject shape with negative dimensions (-1, 1)
Falsifying example: test_ndpointer_negative_shape(
    shape=(-1, 1),  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib

# Test case 1: Create ndpointer with negative shape
ptr = numpy.ctypeslib.ndpointer(shape=(-1, 3))
print(f"Created pointer type: {ptr}")
print(f"Pointer shape attribute: {ptr._shape_}")

# Try to use this pointer with valid NumPy arrays
print("\nAttempting to use pointer with valid arrays:")

# Test with (2, 3) array
arr1 = np.zeros((2, 3))
print(f"Array shape (2, 3): {arr1.shape}")
try:
    ptr.from_param(arr1)
    print("  Accepted")
except TypeError as e:
    print(f"  Error: {e}")

# Test with (1, 3) array
arr2 = np.zeros((1, 3))
print(f"Array shape (1, 3): {arr2.shape}")
try:
    ptr.from_param(arr2)
    print("  Accepted")
except TypeError as e:
    print(f"  Error: {e}")

# Test case 2: Create ndpointer with multiple negative dimensions
ptr2 = numpy.ctypeslib.ndpointer(shape=(0, -1))
print(f"\nCreated second pointer with shape=(0, -1): {ptr2}")
print(f"Pointer shape attribute: {ptr2._shape_}")

# Test case 3: Show that numpy itself rejects negative dimensions
print("\nFor comparison, numpy array creation with negative shape:")
try:
    arr_negative = np.zeros((-1, 3))
    print(f"Created array: {arr_negative}")
except ValueError as e:
    print(f"NumPy error: {e}")
```

<details>

<summary>
ndpointer accepts negative shapes but creates unusable constraints
</summary>
```
Created pointer type: <class 'numpy.ctypeslib._ctypeslib.ndpointer_any_-1x3'>
Pointer shape attribute: (-1, 3)

Attempting to use pointer with valid arrays:
Array shape (2, 3): (2, 3)
  Error: array must have shape (-1, 3)
Array shape (1, 3): (1, 3)
  Error: array must have shape (-1, 3)

Created second pointer with shape=(0, -1): <class 'numpy.ctypeslib._ctypeslib.ndpointer_any_0x-1'>
Pointer shape attribute: (0, -1)

For comparison, numpy array creation with negative shape:
NumPy error: negative dimensions are not allowed
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Inconsistent with NumPy's core behavior**: NumPy itself explicitly rejects negative dimensions when creating arrays, raising `ValueError: negative dimensions are not allowed`. The `ndpointer` function, which is designed to validate NumPy arrays, should follow the same validation rules.

2. **Creates impossible-to-satisfy constraints**: Since NumPy arrays cannot have negative dimensions, an ndpointer with shape `(-1, 3)` can never successfully validate any real NumPy array. This makes the created pointer type completely unusable.

3. **Produces misleading error messages**: When attempting to use the pointer, the error message states "array must have shape (-1, 3)", which is nonsensical and confusing. Users may waste time trying to create an array with this impossible shape.

4. **Violates fail-fast principle**: The invalid input should be rejected at creation time (when `ndpointer` is called), not later when the pointer is used. Early validation would provide clearer error messages and prevent the creation of unusable objects.

5. **Documentation expectation**: While the documentation states that the shape parameter should be a "tuple of ints", it's reasonable to expect these to be non-negative integers, as that's the only valid domain for array shapes in NumPy.

## Relevant Context

- The ndpointer function is located in `/numpy/ctypeslib/_ctypeslib.py` starting at line 239
- The shape parameter processing occurs at lines 314-319, where the shape is converted to a tuple but not validated
- Similar validation is missing for the `ndim` parameter, which also accepts negative values
- Note that zero dimensions ARE valid (e.g., `shape=(0, 5)` correctly matches `np.zeros((0, 5))`)
- The special value `-1` in NumPy's `reshape` function has a specific meaning (infer dimension), but this doesn't apply to ndpointer shapes
- Source code location: `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/ctypeslib/_ctypeslib.py:314-319`

## Proposed Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -313,6 +313,9 @@ def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
     # normalize shape to tuple | None
     if shape is not None:
         try:
             shape = tuple(shape)
+            if any(s < 0 for s in shape):
+                raise ValueError(f"shape dimensions must be non-negative, got {shape}")
         except TypeError:
             # single integer -> 1-tuple
             shape = (shape,)
+            if shape[0] < 0:
+                raise ValueError(f"shape dimensions must be non-negative, got {shape}")
```