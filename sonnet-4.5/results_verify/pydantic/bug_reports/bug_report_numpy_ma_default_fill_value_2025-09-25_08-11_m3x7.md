# Bug Report: numpy.ma Fill Value Functions Fail With Dtype Classes

**Target**: `numpy.ma.default_fill_value`, `numpy.ma.maximum_fill_value`, `numpy.ma.minimum_fill_value`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The fill value functions (`default_fill_value`, `maximum_fill_value`, `minimum_fill_value`) raise `AttributeError` when passed numpy dtype classes (like `np.int32`, `np.float64`) instead of dtype instances, despite NumPy conventionally accepting both forms interchangeably.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings

@given(st.sampled_from([np.int32, np.int64, np.float32, np.float64, np.bool_]))
@settings(max_examples=100)
def test_default_fill_value_accepts_dtype_classes(dtype):
    fill = ma.default_fill_value(dtype)
    assert fill is not None
```

**Failing input**: `np.int32`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

for func_name, func in [('default_fill_value', ma.default_fill_value),
                         ('maximum_fill_value', ma.maximum_fill_value),
                         ('minimum_fill_value', ma.minimum_fill_value)]:
    try:
        fill = func(np.int32)
        print(f"{func_name}(np.int32): {fill}")
    except AttributeError as e:
        print(f"{func_name}(np.int32): FAILS - {e}")

    fill_instance = func(np.dtype('int32'))
    print(f"{func_name}(np.dtype('int32')): {fill_instance}\n")
```

## Why This Is A Bug

NumPy conventionally accepts both dtype classes (`np.int32`) and dtype instances (`np.dtype('int32')`) interchangeably. For example, `np.array([1,2,3], dtype=np.int32)` and `np.array([1,2,3], dtype=np.dtype('int32'))` both work correctly.

All three functions fail with dtype classes because they all use `_get_dtype_of()` which has flawed logic:
- When given `np.int32` (a class), it checks `hasattr(obj, 'dtype')` which returns True
- It then returns `obj.dtype`, which is a getset_descriptor, not a dtype instance
- This descriptor is passed to `_recursive_fill_value()` which tries to access `.names`, causing the AttributeError

## Fix

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -198,6 +198,9 @@ def _get_dtype_of(obj):
     """ Convert the argument for *_fill_value into a dtype """
     if isinstance(obj, np.dtype):
         return obj
+    elif isinstance(obj, type) and issubclass(obj, np.generic):
+        # Handle dtype classes like np.int32, np.float64
+        return np.dtype(obj)
     elif hasattr(obj, 'dtype'):
         return obj.dtype
     else:
```