# Bug Report: numpy.rec.fromrecords Empty List Handling

**Target**: `numpy.rec.fromrecords` and `numpy.rec.fromarrays`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.rec.fromrecords()` and `numpy.rec.fromarrays()` crash with IndexError when given empty input lists, despite regular NumPy arrays handling empty lists correctly.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.rec

@given(st.lists(st.tuples(st.integers(), st.text(max_size=10)), min_size=0, max_size=20))
def test_fromrecords_empty_handling(records):
    rec = numpy.rec.fromrecords(records, names='a,b')
    assert len(rec) == len(records)
```

**Failing input**: `records=[]`

## Reproducing the Bug

```python
import numpy as np
import numpy.rec

numpy.rec.fromrecords([], names='a,b')

numpy.rec.fromarrays([], names='a,b')
```

Both calls raise `IndexError: list index out of range`.

For comparison, regular NumPy arrays handle empty inputs:
```python
np.array([])
```

## Why This Is A Bug

Creating empty structured arrays is a reasonable use case (e.g., initializing a container before populating it). The function crashes instead of returning an empty recarray with the specified dtype, inconsistent with how NumPy handles empty regular arrays.

## Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -621,7 +621,10 @@ def fromarrays(arrayList, dtype=None, shape=None, formats=None,
     # NumPy 1.19.0, 2020-01-01
     shape = _deprecate_shape_0_as_None(shape)

-    if shape is None:
+    if len(arrayList) == 0:
+        shape = (0,)
+        dtype = _dtype_from_descr(dtype, formats, names, titles, aligned, byteorder)
+    elif shape is None:
         shape = arrayList[0].shape
     elif isinstance(shape, int):
         shape = (shape,)
```