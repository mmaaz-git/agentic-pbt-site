# Bug Report: normalize_axis_tuple raises OverflowError instead of AxisError

**Target**: `numpy.lib.array_utils.normalize_axis_tuple`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`normalize_axis_tuple` raises `OverflowError` instead of the documented `AxisError` when passed integer values that exceed the C int range (|value| >= 2^31).

## Property-Based Test

```python
import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple
from hypothesis import given, strategies as st, settings
import pytest


@given(
    axis=st.integers(min_value=2**31, max_value=2**40),
    ndim=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100)
def test_large_positive_axis_raises_axis_error(axis, ndim):
    with pytest.raises(np.exceptions.AxisError):
        normalize_axis_tuple(axis, ndim)
```

**Failing input**: `axis=2_147_483_648, ndim=1`

## Reproducing the Bug

```python
import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

axis = 2_147_483_648
ndim = 1

try:
    normalize_axis_tuple(axis, ndim)
except np.exceptions.AxisError:
    print("AxisError raised (expected)")
except OverflowError:
    print("OverflowError raised (BUG!)")
```

Output:
```
OverflowError raised (BUG!)
```

## Why This Is A Bug

The function's docstring explicitly states:

> Raises
> ------
> AxisError
>     If any axis provided is out of range

However, when an axis value exceeds the C int range (2^31 - 1 for positive or -2^31 for negative), the function raises `OverflowError` instead. This violates the documented API contract and breaks code that expects to catch `AxisError` for all out-of-range axis values.

## Fix

The issue occurs because `normalize_axis_tuple` calls the C-implemented `normalize_axis_index`, which raises `OverflowError` when converting Python ints that don't fit in a C int. The fix is to catch `OverflowError` and convert it to `AxisError`.

```diff
--- a/numpy/_core/numeric.py
+++ b/numpy/_core/numeric.py
@@ -1466,7 +1466,12 @@ def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
         except TypeError:
             pass
     # Going via an iterator directly is slower than via list comprehension.
-    axis = tuple(normalize_axis_index(ax, ndim, argname) for ax in axis)
+    try:
+        axis = tuple(normalize_axis_index(ax, ndim, argname) for ax in axis)
+    except OverflowError:
+        msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
+        if argname:
+            msg = f"{argname}: {msg}"
+        raise np.exceptions.AxisError(msg)
     if not allow_duplicate and len(set(axis)) != len(axis):
         if argname:
             raise ValueError(f'repeated axis in `{argname}` argument')