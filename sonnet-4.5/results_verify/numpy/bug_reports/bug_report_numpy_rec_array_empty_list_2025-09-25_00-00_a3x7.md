# Bug Report: numpy.rec.array Empty List IndexError

**Target**: `numpy.rec.array`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.rec.array()` crashes with `IndexError` when passed an empty list or tuple, attempting to access `obj[0]` without checking if the sequence is empty.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy.rec


@settings(max_examples=500)
@given(st.lists(st.integers(), min_size=0, max_size=20))
def test_array_constructor_preserves_length(int_list):
    try:
        rec_arr = numpy.rec.array(int_list, dtype=[('value', 'i4')])
        assert len(rec_arr) == len(int_list)
    except (ValueError, TypeError):
        pass
```

**Failing input**: `[]`

## Reproducing the Bug

```python
import numpy.rec

numpy.rec.array([], dtype=[('value', 'i4')])
```

Output:
```
IndexError: list index out of range
```

## Why This Is A Bug

1. Empty lists are valid inputs to `numpy.array([])`
2. The function documentation accepts "any" object and doesn't exclude empty sequences
3. The code path for lists/tuples assumes at least one element exists

## Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -1053,7 +1053,10 @@ def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None,
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