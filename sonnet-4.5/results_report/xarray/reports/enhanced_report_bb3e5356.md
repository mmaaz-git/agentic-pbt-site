# Bug Report: xarray.indexes.RangeIndex.linspace Division by Zero with num=1

**Target**: `xarray.indexes.RangeIndex.linspace`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`RangeIndex.linspace` crashes with a `ZeroDivisionError` when called with `num=1` and `endpoint=True` due to an unprotected division by `(num - 1)` at line 283 of range_index.py.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test that discovers the ZeroDivisionError bug in RangeIndex.linspace"""

from hypothesis import given, strategies as st, assume, settings
from xarray.indexes import RangeIndex

@given(
    start=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
    endpoint=st.booleans()
)
@settings(max_examples=1000)
def test_linspace_with_num_1(start, stop, endpoint):
    assume(start != stop)
    index = RangeIndex.linspace(start, stop, num=1, endpoint=endpoint, dim="x")
    assert index.size == 1

if __name__ == "__main__":
    # Run the test to find a failure
    test_linspace_with_num_1()
    print("Test passed!")  # This will not be reached if the test fails
```

<details>

<summary>
**Failing input**: `start=0.0, stop=1.0, endpoint=True`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 20, in <module>
    test_linspace_with_num_1()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 8, in test_linspace_with_num_1
    start=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 15, in test_linspace_with_num_1
    index = RangeIndex.linspace(start, stop, num=1, endpoint=endpoint, dim="x")
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py", line 283, in linspace
    stop += (stop - start) / (num - 1)
            ~~~~~~~~~~~~~~~^~~~~~~~~~~
ZeroDivisionError: float division by zero
Falsifying example: test_linspace_with_num_1(
    start=0.0,  # or any other generated value
    stop=1.0,  # or any other generated value
    endpoint=True,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py:283
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the ZeroDivisionError bug in xarray.indexes.RangeIndex.linspace"""

from xarray.indexes import RangeIndex

# This should create a single-point index at position 0.0
# but instead crashes with ZeroDivisionError
try:
    index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
    print(f"Success! Created index with size {index.size}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError occurred: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
ZeroDivisionError: float division by zero
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/repo.py", line 9, in <module>
    index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py", line 283, in linspace
    stop += (stop - start) / (num - 1)
            ~~~~~~~~~~~~~~~^~~~~~~~~~~
ZeroDivisionError: float division by zero
ZeroDivisionError occurred: float division by zero
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **NumPy Compatibility Violation**: The docstring at line 239-240 explicitly states this method creates "a new RangeIndex from given start / stop values and number of values" and the class documentation at line 119 states the methods are "similar to numpy.linspace". NumPy's `linspace` correctly handles `num=1`:
   - `np.linspace(0.0, 1.0, num=1, endpoint=True)` returns `[0.]`
   - `np.linspace(0.0, 1.0, num=1, endpoint=False)` returns `[0.]`

2. **Mathematical Error**: The code at line 283 performs `stop += (stop - start) / (num - 1)` to adjust the stop value when endpoint=True. This formula calculates the spacing between points, but fails to account for the degenerate case where there's only one point (num=1), causing division by zero.

3. **Inconsistent Behavior**: The function works correctly with `endpoint=False` and `num=1`, but crashes with `endpoint=True`. This inconsistency is unexpected - both cases should produce a single point at the start position.

4. **Valid Use Case**: Single-point coordinates are legitimate in scientific computing for representing degenerate dimensions, placeholder coordinates, or single-sample datasets. The function accepts `num=1` as a valid parameter but fails to handle it correctly.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py` at line 283. The problematic code section:

```python
if endpoint:
    stop += (stop - start) / (num - 1)  # Line 283 - causes ZeroDivisionError when num=1
```

The formula is attempting to adjust the stop value to ensure the last point lands exactly on the original stop value when endpoint=True. For num>1, this makes sense as it stretches the interval. However, when num=1, there's only one point which should be placed at the start position (matching NumPy's behavior), and the division becomes 0/0.

The RangeIndex class is used internally by xarray to create memory-efficient coordinate arrays without materializing all values in memory. This makes the fix important for users working with coordinate systems.

Documentation link: The class is documented at line 110-126 of range_index.py, and the linspace method at lines 227-289.

## Proposed Fix

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -279,8 +279,11 @@ class RangeIndex(CoordinateTransformIndex):
         if coord_name is None:
             coord_name = dim

-        if endpoint:
+        if endpoint and num > 1:
             stop += (stop - start) / (num - 1)
+        elif endpoint and num == 1:
+            # For a single point with endpoint=True, use the start value (matching numpy behavior)
+            stop = start

         transform = RangeCoordinateTransform(
             start, stop, num, coord_name, dim, dtype=dtype
```