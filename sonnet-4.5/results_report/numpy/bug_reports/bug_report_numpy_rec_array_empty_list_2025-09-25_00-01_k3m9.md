# Bug Report: numpy.rec.array Empty List IndexError

**Target**: `numpy.rec.array`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.rec.array([])` crashes with an `IndexError` when passed an empty list, even when a valid dtype is provided.

## Property-Based Test

```python
import numpy.rec as rec
from hypothesis import given, strategies as st
import pytest


@given(st.lists(st.tuples(st.integers(), st.floats(allow_nan=False)), max_size=10))
def test_array_handles_variable_length_lists(records):
    result = rec.array(records, names='a,b')
    assert len(result) == len(records)
```

**Failing input**: `records=[]` (empty list)

## Reproducing the Bug

```python
import numpy.rec as rec

result = rec.array([], dtype=[('x', 'i4')])
```

Output:
```
IndexError: list index out of range
```

## Why This Is A Bug

Empty lists are valid inputs for array construction. NumPy's standard `np.array([])` works correctly, and `rec.array` should handle this case gracefully by returning an empty recarray of the specified dtype.

The bug occurs because the dispatcher in `rec.array()` attempts to inspect `obj[0]` to determine whether to call `fromarrays` or `fromrecords`, without first checking if the list is empty.

## Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -1053,7 +1053,7 @@ def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None,
     elif isinstance(obj, bytes):
         return fromstring(obj, dtype, shape=shape, offset=offset, **kwds)

-    elif isinstance(obj, (list, tuple)):
+    elif isinstance(obj, (list, tuple)) and len(obj) > 0:
         if isinstance(obj[0], (tuple, list)):
             return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
         else:
@@ -1080,6 +1080,11 @@ def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None,
             new = new.copy()
         return new.view(recarray)

+    elif isinstance(obj, (list, tuple)) and len(obj) == 0:
+        # Handle empty list/tuple case
+        if dtype is None:
+            raise ValueError("Must specify dtype for empty list/tuple")
+        return sb.array([], dtype=dtype).view(recarray)
     else:
         interface = getattr(obj, "__array_interface__", None)
         if interface is None or not isinstance(interface, dict):
```