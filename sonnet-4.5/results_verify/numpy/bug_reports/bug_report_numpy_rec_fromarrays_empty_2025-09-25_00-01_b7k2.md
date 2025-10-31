# Bug Report: numpy.rec.fromarrays Empty List IndexError

**Target**: `numpy.rec.fromarrays`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.rec.fromarrays()` crashes with `IndexError` when passed an empty list, attempting to access `arrayList[0].shape` to infer the shape without checking if the list is empty.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import numpy.rec


@settings(max_examples=500)
@given(
    st.lists(st.lists(st.integers(), min_size=1, max_size=5), min_size=0, max_size=3),
    st.lists(st.text(alphabet='abc', min_size=1, max_size=3), min_size=1, max_size=5)
)
def test_fromarrays_round_trip(int_arrays, names_list):
    if len(int_arrays) != len(names_list):
        return

    arrays = [np.array(arr) for arr in int_arrays]
    names = ','.join(names_list)

    try:
        rec_arr = numpy.rec.fromarrays(arrays, names=names)
        assert len(rec_arr) == (len(arrays[0]) if arrays else 0)
    except (ValueError, TypeError):
        pass
```

**Failing input**: `int_arrays=[], names_list=['a']`

## Reproducing the Bug

```python
import numpy.rec

numpy.rec.fromarrays([], names='a')
```

Output:
```
IndexError: list index out of range
```

## Why This Is A Bug

1. Empty lists are valid inputs to array constructors
2. The function claims to create a record array from a "list of array-like objects" - an empty list is a valid list
3. The code assumes at least one array exists to infer the shape from

## Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -621,7 +621,7 @@ def fromarrays(arrayList, dtype=None, shape=None, formats=None,
     shape = _deprecate_shape_0_as_None(shape)

     if shape is None:
-        shape = arrayList[0].shape
+        shape = arrayList[0].shape if arrayList else (0,)
     elif isinstance(shape, int):
         shape = (shape,)
```