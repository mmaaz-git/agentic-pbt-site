# Bug Report: xarray RangeIndex.linspace Division by Zero Error

**Target**: `xarray.indexes.range_index.RangeIndex.linspace`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `RangeIndex.linspace` method crashes with a `ZeroDivisionError` when called with `num=1` and `endpoint=True` due to an unchecked division by zero in the endpoint adjustment calculation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.indexes.range_index import RangeIndex

@given(
    start=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    stop=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    num=st.integers(min_value=1, max_value=100),
    endpoint=st.booleans()
)
@settings(max_examples=1000)
def test_linspace_no_crash(start, stop, num, endpoint):
    index = RangeIndex.linspace(start, stop, num=num, endpoint=endpoint, dim="x")
    assert index.size == num

if __name__ == "__main__":
    test_linspace_no_crash()
```

<details>

<summary>
**Failing input**: `start=0.0, stop=0.0, num=1, endpoint=True`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 16, in <module>
    test_linspace_no_crash()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 5, in test_linspace_no_crash
    start=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 12, in test_linspace_no_crash
    index = RangeIndex.linspace(start, stop, num=num, endpoint=endpoint, dim="x")
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py", line 283, in linspace
    stop += (stop - start) / (num - 1)
            ~~~~~~~~~~~~~~~^~~~~~~~~~~
ZeroDivisionError: float division by zero
Falsifying example: test_linspace_no_crash(
    start=0.0,  # or any other generated value
    stop=0.0,  # or any other generated value
    num=1,
    endpoint=True,
)
```
</details>

## Reproducing the Bug

```python
from xarray.indexes.range_index import RangeIndex

# This should work like numpy.linspace(0.0, 1.0, num=1, endpoint=True)
# which returns array([0.])
index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
```

<details>

<summary>
ZeroDivisionError: float division by zero
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/repo.py", line 5, in <module>
    index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py", line 283, in linspace
    stop += (stop - start) / (num - 1)
            ~~~~~~~~~~~~~~~^~~~~~~~~~~
ZeroDivisionError: float division by zero
```
</details>

## Why This Is A Bug

The method's docstring states it is "similar to numpy.linspace" (line 119-120 in range_index.py), creating an expectation that it should handle the same inputs that numpy does. NumPy's `linspace` handles `num=1` with `endpoint=True` gracefully, returning an array containing the start value. For example, `np.linspace(0.0, 1.0, num=1, endpoint=True)` returns `[0.]`.

The crash occurs on line 283 of range_index.py where the code attempts to adjust the stop value using the formula `stop += (stop - start) / (num - 1)`. When `num=1`, the expression `(num - 1)` evaluates to 0, causing division by zero. This is a mathematical edge case that should have been handled, as the adjustment is only needed when there are multiple points to space evenly.

The method already works correctly with `num=1` when `endpoint=False`, demonstrating that single-point ranges are intended to be supported. The crash only occurs with the specific combination of `num=1` and `endpoint=True`, suggesting an oversight in the implementation rather than a deliberate design choice.

## Relevant Context

The `RangeIndex` class is designed to be a memory-efficient index for evenly-spaced floating-point values. The `linspace` method is one of two factory methods (along with `arange`) for creating these indices. The documentation explicitly compares this method to numpy.linspace, and users would reasonably expect similar behavior.

The mathematical logic behind the endpoint adjustment is to ensure proper spacing when the endpoint is included. With multiple points, the interval needs to be extended slightly to ensure the last point falls exactly on the specified stop value. However, when there's only one point (`num=1`), there's no spacing to calculate - the single point should simply be the start value, matching numpy's behavior.

Code location: `/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py:283`

## Proposed Fix

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -280,7 +280,10 @@ class RangeIndex(CoordinateTransformIndex):
         if coord_name is None:
             coord_name = dim

-        if endpoint:
+        if endpoint and num == 1:
+            # Special case: single point, use start value (matching numpy behavior)
+            # No adjustment needed to stop value
+        elif endpoint:
             stop += (stop - start) / (num - 1)

         transform = RangeCoordinateTransform(
```