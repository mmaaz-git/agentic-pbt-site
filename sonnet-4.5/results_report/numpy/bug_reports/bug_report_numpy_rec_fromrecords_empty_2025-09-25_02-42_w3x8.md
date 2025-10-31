# Bug Report: numpy.rec.fromrecords Empty List Crash

**Target**: `numpy.rec.fromrecords`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.rec.fromrecords` crashes with `IndexError` when given an empty list, despite empty record arrays being valid in NumPy.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np

@given(st.lists(st.tuples(st.integers(), st.floats(allow_nan=False, allow_infinity=False)), min_size=0, max_size=10))
def test_fromrecords_length_invariant(records):
    rec = np.rec.fromrecords(records, names='x,y')
    assert len(rec) == len(records)
```

**Failing input**: `records=[]`

## Reproducing the Bug

```python
import numpy as np

rec = np.rec.fromrecords([], names='x,y')
```

This crashes with:
```
IndexError: list index out of range
```

However, empty record arrays are valid:
```python
np.recarray(shape=(0,), dtype=[('x', int), ('y', int)])
```

And the related function `fromarrays` handles empty input correctly:
```python
np.rec.fromarrays([[], []], names='a,b')
```

## Why This Is A Bug

1. Empty collections are a standard edge case that should be handled
2. NumPy explicitly supports empty structured arrays
3. The related function `fromarrays` handles empty input correctly
4. Users would reasonably expect to pass an empty list to `fromrecords`
5. The crash occurs because `fromarrays` (called internally) tries to access `arrayList[0].shape` without checking if the list is empty

## Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -621,7 +621,7 @@ def fromarrays(arrayList, dtype=None, shape=None, formats=None,
     # NumPy 1.19.0, 2020-01-01
     shape = _deprecate_shape_0_as_None(shape)

-    if shape is None:
+    if shape is None and len(arrayList) > 0:
         shape = arrayList[0].shape
     elif shape is not None:
         shape = (shape,)
```

Or more robustly, set `shape = (0,)` when `arrayList` is empty:

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -621,7 +621,10 @@ def fromarrays(arrayList, dtype=None, shape=None, formats=None,
     # NumPy 1.19.0, 2020-01-01
     shape = _deprecate_shape_0_as_None(shape)

-    if shape is None:
+    if shape is None and len(arrayList) == 0:
+        shape = (0,)
+    elif shape is None:
         shape = arrayList[0].shape
     elif shape is not None:
         shape = (shape,)
```
