# Bug Report: numpy.rec.fromrecords IndexError on Empty List

**Target**: `numpy.rec.fromrecords`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.rec.fromrecords([])` crashes with IndexError when called with an empty list, but should return an empty recarray.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.rec as rec

@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=0, max_size=10))
def test_fromrecords_empty_list(records):
    if len(records) == 0:
        r = rec.fromrecords(records, names='x,y')
        assert len(r) == 0
    else:
        r = rec.fromrecords(records, names='x,y')
        assert len(r) == len(records)
```

**Failing input**: `records=[]`

## Reproducing the Bug

```python
import numpy.rec as rec

r = rec.fromrecords([], names='x,y')
```

## Why This Is A Bug

Creating record arrays from empty data is a reasonable use case (e.g., filtering operations that may return no results). NumPy generally supports empty arrays, so `fromrecords` should handle empty input gracefully and return an empty recarray rather than crashing.

## Fix

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
```