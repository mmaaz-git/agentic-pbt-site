# Bug Report: numpy.rec.array Empty List Crash

**Target**: `numpy.rec.array`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.rec.array` crashes with `IndexError` when passed an empty list, instead of handling it gracefully or returning an empty record array.

## Property-Based Test

```python
import numpy.rec
from hypothesis import given, strategies as st


@given(st.lists(st.integers(), min_size=0, max_size=30))
def test_array_handles_all_list_sizes(lst):
    result = numpy.rec.array(lst)
    assert isinstance(result, numpy.rec.recarray)
```

**Failing input**: `[]`

## Reproducing the Bug

```python
import numpy.rec

result = numpy.rec.array([])
```

Output:
```
IndexError: list index out of range
  File "numpy/_core/records.py", line 1056, in array
    if isinstance(obj[0], (tuple, list)):
```

## Why This Is A Bug

The function is documented as a "general-purpose record array constructor" that should handle a "wide-variety of objects". When a user passes an empty list, the function attempts to inspect the first element (`obj[0]`) to determine whether to dispatch to `fromarrays` or `fromrecords`, but fails to check if the list is empty first. This causes a crash on valid input (an empty list is a perfectly valid list object).

Empty arrays and lists are common in data processing pipelines, especially during filtering or initial states, so this crash would affect real users.

## Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -1053,7 +1053,10 @@ def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None,
     elif isinstance(obj, bytes):
         return fromstring(obj, dtype, shape=shape, offset=offset, **kwds)

     elif isinstance(obj, (list, tuple)):
+        if len(obj) == 0:
+            return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
+
         if isinstance(obj[0], (tuple, list)):
             return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
         else:
```