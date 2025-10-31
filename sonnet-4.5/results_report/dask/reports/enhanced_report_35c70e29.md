# Bug Report: dask.array.squeeze Raises IndexError Instead of AxisError for Out-of-Bounds Axis

**Target**: `dask.array.squeeze`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`dask.array.squeeze` raises an uninformative `IndexError` instead of the expected `AxisError` when given an out-of-bounds axis parameter, breaking API compatibility with NumPy's squeeze function.

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

if __name__ == "__main__":
    test_squeeze_raises_correct_error_for_invalid_axis()
```

<details>

<summary>
**Failing input**: `shape=[1], axis=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 39, in <module>
    test_squeeze_raises_correct_error_for_invalid_axis()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 10, in test_squeeze_raises_correct_error_for_invalid_axis
    st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=3),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 35, in test_squeeze_raises_correct_error_for_invalid_axis
    assert np_error == da_error or (np_error == 'AxisError' and da_error in ['AxisError', 'ValueError']), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: NumPy raises AxisError, but Dask raises IndexError for shape=[1], axis=1
Falsifying example: test_squeeze_raises_correct_error_for_invalid_axis(
    shape=[1],
    axis=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/21/hypo.py:18
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import numpy as np
import dask.array as da

shape = [1]
axis = 1

x_np = np.random.rand(*shape)
x_da = da.from_array(x_np, chunks='auto')

print("Testing squeeze with shape=[1], axis=1 (out of bounds)")
print("=" * 60)

try:
    result = np.squeeze(x_np, axis=axis)
    print("NumPy: No error raised (unexpected)")
except Exception as e:
    print(f"NumPy error: {type(e).__name__}: {e}")

try:
    result = da.squeeze(x_da, axis=axis).compute()
    print("Dask: No error raised (unexpected)")
except Exception as e:
    print(f"Dask error: {type(e).__name__}: {e}")
```

<details>

<summary>
Error comparison: NumPy raises descriptive AxisError, Dask raises generic IndexError
</summary>
```
Testing squeeze with shape=[1], axis=1 (out of bounds)
============================================================
NumPy error: AxisError: axis 1 is out of bounds for array of dimension 1
Dask error: IndexError: tuple index out of range
```
</details>

## Why This Is A Bug

This violates Dask's documented NumPy API compatibility in three important ways:

1. **Incorrect error type**: NumPy raises `AxisError` (from `numpy.exceptions`), a specialized error that clearly indicates an axis-related problem. Dask raises `IndexError`, a generic Python error that provides no context about what went wrong.

2. **Poor error message**: NumPy's error message ("axis 1 is out of bounds for array of dimension 1") immediately tells users what went wrong. Dask's message ("tuple index out of range") is cryptic and doesn't indicate that the axis parameter was invalid.

3. **Inconsistent with Dask's own validation**: Dask already has a `validate_axis` function that would raise the proper `AxisError` with a descriptive message, but it's called too late in the function execution.

The bug occurs because the squeeze function checks if dimensions equal 1 (line 1949) before validating that the axis indices are valid (line 1952). This ordering causes Python to raise an IndexError when accessing `a.shape[i]` with an out-of-bounds index.

## Relevant Context

The issue is located in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/array/routines.py:1949`. The problematic code structure is:

```python
def squeeze(a, axis=None):
    # ... axis normalization ...

    # This line tries to access a.shape[i] before validating i is in bounds
    if any(a.shape[i] != 1 for i in axis):  # Line 1949 - IndexError here
        raise ValueError("cannot squeeze axis with size other than one")

    # This validation should come first
    axis = validate_axis(axis, a.ndim)  # Line 1952
```

The `validate_axis` function (in `dask/array/utils.py`) correctly implements axis validation:
- Checks if axis is an integer
- Validates bounds: `-ndim <= axis < ndim`
- Raises `AxisError` with a descriptive message for out-of-bounds values
- Handles negative indexing properly

This is particularly important because:
- Dask's squeeze function is decorated with `@derived_from(np)`, explicitly claiming NumPy compatibility
- Users migrating from NumPy to Dask expect consistent error behavior for debugging
- The fix is trivial - just reorder two lines of code

## Proposed Fix

```diff
--- a/dask/array/routines.py
+++ b/dask/array/routines.py
@@ -1946,10 +1946,10 @@ def squeeze(a, axis=None):
     elif not isinstance(axis, tuple):
         axis = (axis,)

+    axis = validate_axis(axis, a.ndim)
+
     if any(a.shape[i] != 1 for i in axis):
         raise ValueError("cannot squeeze axis with size other than one")

-    axis = validate_axis(axis, a.ndim)
-
     sl = tuple(0 if i in axis else slice(None) for i, s in enumerate(a.shape))
```