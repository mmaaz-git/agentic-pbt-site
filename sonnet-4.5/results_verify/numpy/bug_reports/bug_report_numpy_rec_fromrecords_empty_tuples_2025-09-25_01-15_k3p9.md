# Bug Report: numpy.rec.fromrecords Empty Tuples Crash

**Target**: `numpy.rec.fromrecords`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.rec.fromrecords` crashes with `IndexError: list index out of range` when given a list of empty tuples, even though NumPy supports structured arrays with no fields.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.rec


@given(st.lists(st.tuples(), min_size=1, max_size=10))
def test_fromrecords_empty_tuples(records):
    rec_arr = numpy.rec.fromrecords(records)
    assert len(rec_arr) == len(records)
```

**Failing input**: `[()]` (or any non-empty list of empty tuples)

## Reproducing the Bug

```python
import numpy.rec

records = [(), (), ()]
rec_arr = numpy.rec.fromrecords(records)
```

**Output**:
```
IndexError: list index out of range
```

**Full traceback**:
```
  File "numpy/_core/records.py", line 713, in fromrecords
    return fromarrays(arrlist, formats=formats, shape=shape, names=names,
  File "numpy/_core/records.py", line 624, in fromarrays
    shape = arrayList[0].shape
            ~~~~~~~~~^^^
IndexError: list index out of range
```

## Why This Is A Bug

1. **NumPy supports empty structured types**: `np.zeros(5, dtype=[])` successfully creates an array of 5 empty records
2. **Inconsistent behavior**: `numpy.rec.fromarrays([])` works (creates empty array), but `numpy.rec.fromrecords([(), ()])` crashes
3. **Poor error message**: If empty tuples aren't supported, the error should be a clear `ValueError` with a helpful message, not an `IndexError` from accessing a missing list element
4. **Valid use case**: Empty records represent data with no fields, which is a valid (if rare) use case

## Fix

The bug is in `fromarrays` at line 624 in `numpy/_core/records.py`. When `arrayList` is empty (which happens when `fromrecords` receives empty tuples), the code tries to access `arrayList[0]` without checking if the list is non-empty.

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

This fix allows empty record arrays to be created, consistent with NumPy's support for empty structured types.