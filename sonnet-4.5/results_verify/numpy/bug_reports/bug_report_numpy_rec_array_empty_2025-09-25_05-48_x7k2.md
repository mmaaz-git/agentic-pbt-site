# Bug Report: numpy.rec.array IndexError on Empty List/Tuple

**Target**: `numpy.rec.array`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.rec.array([])` and `numpy.rec.array(())` crash with IndexError when called with an empty list or tuple, but should return an empty recarray.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.rec as rec
import pytest

@given(st.lists(st.lists(st.integers(), min_size=2, max_size=2), min_size=0, max_size=10))
def test_array_handles_empty_input(records):
    records_tuples = [tuple(r) for r in records]
    r = rec.array(records_tuples, formats=['i4', 'i4'], names='x,y')
    assert len(r) == len(records)
```

**Failing input**: `records=[]`

## Reproducing the Bug

```python
import numpy.rec as rec

r = rec.array([], formats=['i4'], names='x')
```

## Why This Is A Bug

Creating record arrays from empty data is a reasonable use case (e.g., filtering operations that may return no results). The `array()` function is a general-purpose constructor that should handle empty inputs gracefully, especially since other numpy functions support empty arrays.

## Fix

This bug is caused by the same underlying issue as the `fromrecords([])` bug. Both need fixes to handle empty inputs:

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -621,7 +621,10 @@ def fromarrays(arrayList, dtype=None, shape=None, formats=None,
     shape = _deprecate_shape_0_as_None(shape)

     if shape is None:
-        shape = arrayList[0].shape
+        if len(arrayList) == 0:
+            shape = (0,)
+        else:
+            shape = arrayList[0].shape
     elif isinstance(shape, int):
         shape = (shape,)

@@ -1053,7 +1056,10 @@ def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None,
         return fromstring(obj, dtype, shape=shape, offset=offset, **kwds)

     elif isinstance(obj, (list, tuple)):
-        if isinstance(obj[0], (tuple, list)):
+        if len(obj) == 0:
+            return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
+        elif isinstance(obj[0], (tuple, list)):
             return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
         else:
             return fromarrays(obj, dtype=dtype, shape=shape, **kwds)
```