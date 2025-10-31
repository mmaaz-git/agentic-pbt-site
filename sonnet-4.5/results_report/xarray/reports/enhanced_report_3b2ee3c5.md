# Bug Report: xarray.indexes.RangeIndex.linspace Division by Zero with num=1

**Target**: `xarray.indexes.RangeIndex.linspace`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`RangeIndex.linspace` crashes with a `ZeroDivisionError` when called with `num=1` and `endpoint=True`, failing to handle a mathematically valid case that NumPy's linspace handles correctly.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for RangeIndex.linspace with num=1 and endpoint=True"""

from hypothesis import given, strategies as st, example
from xarray.indexes import RangeIndex


@given(
    start=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    stop=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
)
@example(start=0.0, stop=0.0)  # Explicitly test with both values equal
@example(start=0.0, stop=1.0)  # Standard case
def test_linspace_num_1_endpoint_true(start, stop):
    """Test that linspace works with num=1 and endpoint=True."""
    idx = RangeIndex.linspace(start, stop, num=1, endpoint=True, dim="x")
    assert idx.size == 1

    # Also test that we get a reasonable coordinate value
    coord_values = idx.transform.forward({idx.dim: [0]})
    assert idx.coord_name in coord_values
    assert len(coord_values[idx.coord_name]) == 1


if __name__ == "__main__":
    test_linspace_num_1_endpoint_true()
```

<details>

<summary>
**Failing input**: `start=0.0, stop=0.0` and `start=0.0, stop=1.0`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 26, in <module>
  |     test_linspace_num_1_endpoint_true()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 9, in test_linspace_num_1_endpoint_true
  |     start=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures in explicit examples. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 16, in test_linspace_num_1_endpoint_true
    |     idx = RangeIndex.linspace(start, stop, num=1, endpoint=True, dim="x")
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py", line 283, in linspace
    |     stop += (stop - start) / (num - 1)
    |             ~~~~~~~~~~~~~~~^~~~~~~~~~~
    | ZeroDivisionError: float division by zero
    | Falsifying explicit example: test_linspace_num_1_endpoint_true(
    |     start=0.0,
    |     stop=0.0,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 16, in test_linspace_num_1_endpoint_true
    |     idx = RangeIndex.linspace(start, stop, num=1, endpoint=True, dim="x")
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py", line 283, in linspace
    |     stop += (stop - start) / (num - 1)
    |             ~~~~~~~~~~~~~~~^~~~~~~~~~~
    | ZeroDivisionError: float division by zero
    | Falsifying explicit example: test_linspace_num_1_endpoint_true(
    |     start=0.0,
    |     stop=1.0,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the RangeIndex.linspace bug with num=1"""

import sys
import traceback
import numpy as np

# First show that NumPy handles this case correctly
print("NumPy's behavior with num=1 and endpoint=True:")
print("np.linspace(0.0, 1.0, num=1, endpoint=True):", np.linspace(0.0, 1.0, num=1, endpoint=True))
print("np.linspace(5.0, 10.0, num=1, endpoint=True):", np.linspace(5.0, 10.0, num=1, endpoint=True))
print()

# Now try xarray's RangeIndex.linspace
print("Attempting xarray.indexes.RangeIndex.linspace with num=1 and endpoint=True:")
print("Code: RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim='x')")
print()

try:
    from xarray.indexes import RangeIndex
    idx = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim='x')
    print("Success! Result:", idx)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    print()
    print("Full traceback:")
    traceback.print_exc()
```

<details>

<summary>
ZeroDivisionError when creating single-point RangeIndex
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/repo.py", line 21, in <module>
    idx = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim='x')
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py", line 283, in linspace
    stop += (stop - start) / (num - 1)
            ~~~~~~~~~~~~~~~^~~~~~~~~~~
ZeroDivisionError: float division by zero
NumPy's behavior with num=1 and endpoint=True:
np.linspace(0.0, 1.0, num=1, endpoint=True): [0.]
np.linspace(5.0, 10.0, num=1, endpoint=True): [5.]

Attempting xarray.indexes.RangeIndex.linspace with num=1 and endpoint=True:
Code: RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim='x')

ERROR: ZeroDivisionError: float division by zero

Full traceback:

```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Valid Input Rejected**: The function accepts `num` as a positive integer parameter with no documented minimum value restriction, yet crashes on the valid input `num=1`.

2. **NumPy Compatibility**: The function is named `linspace` and has the same signature as NumPy's `linspace`, creating a strong expectation of similar behavior. NumPy correctly handles `num=1` by returning an array with the start value.

3. **Mathematical Validity**: Creating a single-point coordinate array is a mathematically valid and common operation in scientific computing. Single-element dimensions are frequently used in data analysis.

4. **Unhandled Exception**: The function fails with an unhandled `ZeroDivisionError` rather than gracefully handling this edge case or providing a meaningful error message.

5. **Documentation Gap**: The docstring states the function creates "evenly spaced, monotonic floating-point values" but doesn't specify that `num` must be greater than 1. The parameter is documented simply as "Number of values in the interval."

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py` at line 283:

```python
if endpoint:
    stop += (stop - start) / (num - 1)  # Division by zero when num=1
```

The code attempts to adjust the `stop` value when `endpoint=True` to ensure the final value equals the original stop value. However, it assumes `num > 1` without validation.

For comparison, NumPy's implementation handles this case appropriately. When `num=1` with `endpoint=True`, NumPy returns an array containing just the start value, which is the mathematically sensible result for a single-point linear space.

**Documentation reference**: https://docs.xarray.dev/en/stable/generated/xarray.indexes.RangeIndex.linspace.html

**Code location**: xarray/indexes/range_index.py:283

## Proposed Fix

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -279,8 +279,12 @@ class RangeIndex(CoordinateTransformIndex):
         if coord_name is None:
             coord_name = dim

-        if endpoint:
+        if endpoint and num > 1:
             stop += (stop - start) / (num - 1)
+        elif endpoint and num == 1:
+            # When num=1 with endpoint=True, the single point should be at start
+            # (matching NumPy's behavior)
+            stop = start

         transform = RangeCoordinateTransform(
             start, stop, num, coord_name, dim, dtype=dtype
```