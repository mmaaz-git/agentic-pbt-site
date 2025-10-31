# Bug Report: numpy.ctypeslib.ndpointer Shape Validation

**Target**: `numpy.ctypeslib.ndpointer`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ndpointer` function accepts invalid shape types (strings, dicts, sets) instead of validating that shape is an integer or tuple of integers, leading to confusing validation errors.

## Property-Based Test

```python
import numpy as np
import numpy.ctypeslib as npc
from hypothesis import given, strategies as st, settings

@given(shape_str=st.text(min_size=1, max_size=20))
@settings(max_examples=300)
def test_ndpointer_rejects_string_shape(shape_str):
    try:
        ptr = npc.ndpointer(shape=shape_str)
        assert False, f"ndpointer should reject string shape, but got _shape_={ptr._shape_}"
    except (TypeError, ValueError) as e:
        pass
```

**Failing input**: `shape='0'`

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib as npc

ptr_str = npc.ndpointer(shape="abc")
print(f"String shape 'abc' -> _shape_ = {ptr_str._shape_}")

ptr_dict = npc.ndpointer(shape={"x": 1, "y": 2})
print(f"Dict shape -> _shape_ = {ptr_dict._shape_}")

arr = np.zeros((3,), dtype=np.int32)
try:
    ptr_str.from_param(arr)
except TypeError as e:
    print(f"Confusing error: {e}")
```

Expected output:
```
TypeError: shape must be an integer or tuple of integers
```

Actual output:
```
String shape 'abc' -> _shape_ = ('a', 'b', 'c')
Dict shape -> _shape_ = ('x', 'y')
Confusing error: array must have shape ('a', 'b', 'c')
```

## Why This Is A Bug

The documentation states that `shape` should be "tuple of ints", but the function silently accepts any iterable and converts it via `tuple()`. This violates the API contract and produces confusing error messages when users accidentally pass wrong types. For example, passing a string "100" creates a shape of ('1', '0', '0'), which is nonsensical.

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -290,6 +290,13 @@ def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
     # normalize shape to tuple | None
     if shape is not None:
         try:
             shape = tuple(shape)
+            # Validate that all elements are integers
+            if not all(isinstance(s, (int, np.integer)) for s in shape):
+                raise TypeError(
+                    f"shape must be an integer or tuple of integers, "
+                    f"not {type(shape).__name__} with elements {shape}"
+                )
         except TypeError:
             # single integer -> 1-tuple
+            if not isinstance(shape, (int, np.integer)):
+                raise TypeError(
+                    f"shape must be an integer or tuple of integers, "
+                    f"not {type(shape).__name__}"
+                )
             shape = (shape,)
```