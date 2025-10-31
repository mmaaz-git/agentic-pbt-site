# Bug Report: numpy.ctypeslib.ndpointer ndim Type Validation

**Target**: `numpy.ctypeslib.ndpointer`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`ndpointer` accepts non-integer `ndim` values (floats, inf, nan) without validation, causing crashes or incorrect behavior.

## Property-Based Test

```python
import numpy as np
import numpy.ctypeslib as npc
from hypothesis import given, strategies as st, settings, assume

@given(ndim_value=st.one_of(st.floats(), st.text(), st.lists(st.integers())))
@settings(max_examples=200)
def test_ndpointer_ndim_type_validation(ndim_value):
    assume(not isinstance(ndim_value, (int, type(None))))

    try:
        ptr = npc.ndpointer(ndim=ndim_value)
        arr = np.zeros((2, 3), dtype=np.int32)
        result = ptr.from_param(arr)
        assert False, f"Should reject non-integer ndim: {ndim_value}"
    except (TypeError, ValueError, OverflowError) as e:
        pass
```

**Failing input**: `ndim=float('inf')`

## Reproducing the Bug

```python
import numpy as np
import numpy.ctypeslib as npc

ptr1 = npc.ndpointer(ndim=2.5)
print(f"Float ndim accepted: _ndim_ = {ptr1._ndim_}")

ptr2 = npc.ndpointer(ndim=float('inf'))
```

Output:
```
Float ndim accepted: _ndim_ = 2.5
Traceback (most recent call last):
  ...
OverflowError: cannot convert float infinity to integer
```

## Why This Is A Bug

The documentation specifies `ndim` as "int, optional", but the function accepts any value without validation. This leads to:
1. Incorrect validation when non-integer ndim is stored (e.g., 2.5)
2. OverflowError when ndim is infinity (crashes during name generation)
3. ValueError when ndim is NaN

## Fix

```diff
--- a/numpy/ctypeslib/_ctypeslib.py
+++ b/numpy/ctypeslib/_ctypeslib.py
@@ -280,6 +280,10 @@ def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
     # normalize dtype to dtype | None
     if dtype is not None:
         dtype = np.dtype(dtype)

+    # validate ndim
+    if ndim is not None and not isinstance(ndim, (int, np.integer)):
+        raise TypeError(f"ndim must be an integer, not {type(ndim).__name__}")
+
     # normalize flags to int | None
     num = None
     if flags is not None:
```