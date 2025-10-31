# Bug Report: dask.array.squeeze IndexError on Invalid Axis

**Target**: `dask.array.squeeze`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`dask.array.squeeze` raises `IndexError` instead of `AxisError` (or `ValueError`) when called with an out-of-bounds axis parameter, breaking compatibility with NumPy's behavior.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import numpy as np
import dask.array as da
from hypothesis import given, strategies as st, settings


@given(
    st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=3),
    st.integers(min_value=0, max_value=5)
)
@settings(max_examples=200)
def test_squeeze_raises_correct_error_for_invalid_axis(shape, axis):
    if axis < len(shape):
        return

    x_np = np.random.rand(*shape)
    x_da = da.from_array(x_np, chunks='auto')

    np_error = None
    da_error = None

    try:
        np.squeeze(x_np, axis=axis)
    except Exception as e:
        np_error = type(e).__name__

    try:
        da.squeeze(x_da, axis=axis).compute()
    except Exception as e:
        da_error = type(e).__name__

    if np_error is not None and da_error is not None:
        assert np_error == da_error or (np_error == 'AxisError' and da_error in ['AxisError', 'ValueError']), \
            f"NumPy raises {np_error}, but Dask raises {da_error} for shape={shape}, axis={axis}"
```

**Failing input**: `shape=[1], axis=1`

## Reproducing the Bug

```python
import numpy as np
import dask.array as da

shape = [1]
axis = 1

x_np = np.random.rand(*shape)
x_da = da.from_array(x_np, chunks='auto')

try:
    np.squeeze(x_np, axis=axis)
except Exception as e:
    print(f"NumPy error: {type(e).__name__}: {e}")

try:
    da.squeeze(x_da, axis=axis).compute()
except Exception as e:
    print(f"Dask error: {type(e).__name__}: {e}")
```

**Output:**
```
NumPy error: AxisError: axis 1 is out of bounds for array of dimension 1
Dask error: IndexError: tuple index out of range
```

## Why This Is A Bug

Dask aims to provide a NumPy-compatible API. When an invalid axis is provided to `squeeze`, NumPy raises an `AxisError` (or `np.AxisError` in older versions) with a clear message. Dask instead raises an `IndexError` with a less informative message, breaking API compatibility.

The issue occurs in `dask/array/routines.py` where the code checks if dimensions are of size 1 before validating that the axis indices are within bounds:

```python
def squeeze(a, axis=None):
    if axis is None:
        axis = tuple(i for i, d in enumerate(a.shape) if d == 1)
    elif not isinstance(axis, tuple):
        axis = (axis,)

    if any(a.shape[i] != 1 for i in axis):  # Line 8: IndexError happens here
        raise ValueError("cannot squeeze axis with size other than one")

    axis = validate_axis(axis, a.ndim)  # Line 11: This should come first
    ...
```

## Fix

```diff
--- a/dask/array/routines.py
+++ b/dask/array/routines.py
@@ -5,10 +5,10 @@ def squeeze(a, axis=None):
     elif not isinstance(axis, tuple):
         axis = (axis,)

+    axis = validate_axis(axis, a.ndim)
+
     if any(a.shape[i] != 1 for i in axis):
         raise ValueError("cannot squeeze axis with size other than one")

-    axis = validate_axis(axis, a.ndim)
-
     sl = tuple(0 if i in axis else slice(None) for i, s in enumerate(a.shape))

     # Return 0d Dask Array if all axes are squeezed,
```

The fix moves the `validate_axis` call before the shape check, ensuring that out-of-bounds axis indices are caught with the appropriate error type before attempting to access `a.shape[i]`.