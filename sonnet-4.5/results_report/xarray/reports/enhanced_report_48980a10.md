# Bug Report: xarray.indexes.RangeIndex.linspace Division by Zero with num=1

**Target**: `xarray.indexes.RangeIndex.linspace`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `RangeIndex.linspace` method crashes with a `ZeroDivisionError` when called with `num=1` and `endpoint=True`, instead of returning a single-element range like NumPy's linspace does.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from xarray.indexes import RangeIndex
import numpy as np

@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=200)
def test_linspace_num_one_endpoint(start, stop):
    assume(start != stop)

    index = RangeIndex.linspace(start, stop, num=1, endpoint=True, dim="x")

    transform = index.transform
    coords = transform.forward({transform.dim: np.array([0])})
    values = coords[transform.coord_name]

    assert len(values) == 1

if __name__ == "__main__":
    test_linspace_num_one_endpoint()
```

<details>

<summary>
**Failing input**: `start=0.0, stop=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 22, in <module>
    test_linspace_num_one_endpoint()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 6, in test_linspace_num_one_endpoint
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 13, in test_linspace_num_one_endpoint
    index = RangeIndex.linspace(start, stop, num=1, endpoint=True, dim="x")
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py", line 283, in linspace
    stop += (stop - start) / (num - 1)
            ~~~~~~~~~~~~~~~^~~~~~~~~~~
ZeroDivisionError: float division by zero
Falsifying example: test_linspace_num_one_endpoint(
    # The test always failed when commented parts were varied together.
    start=0.0,  # or any other generated value
    stop=1.0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from xarray.indexes import RangeIndex

# This should demonstrate the ZeroDivisionError
index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
print(f"Index created: {index}")
```

<details>

<summary>
ZeroDivisionError at line 283 of range_index.py
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/repo.py", line 4, in <module>
    index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py", line 283, in linspace
    stop += (stop - start) / (num - 1)
            ~~~~~~~~~~~~~~~^~~~~~~~~~~
ZeroDivisionError: float division by zero
```
</details>

## Why This Is A Bug

The `RangeIndex.linspace` method is designed to emulate NumPy's `linspace` function, as evidenced by its name, parameter signature, and documentation. NumPy's `linspace` handles the edge case of `num=1` gracefully by returning an array containing just the start value, regardless of the `endpoint` parameter setting. For example, `numpy.linspace(0, 1, 1, endpoint=True)` returns `[0.]` without error.

The xarray implementation fails to handle this case due to the code at line 283 in `/xarray/indexes/range_index.py`:
```python
if endpoint:
    stop += (stop - start) / (num - 1)
```
When `num=1`, this performs division by zero (`num - 1 = 0`), causing the crash. The documentation for this method states it creates "evenly spaced values" with `num` being the "Number of values in the interval", which implies `num=1` should be valid input that returns a single value.

## Relevant Context

The `RangeIndex.linspace` method is documented to accept a `num` parameter representing the "Number of values in the interval, i.e., dimension size (default: 50)". There are no documented constraints that `num` must be greater than 1. The method's naming and design clearly reference NumPy's `linspace` function, which sets user expectations for similar behavior.

NumPy's behavior with `num=1`:
- `numpy.linspace(0, 1, 1, endpoint=True)` returns `[0.]`
- `numpy.linspace(0, 1, 1, endpoint=False)` returns `[0.]`
- With only one point, the endpoint parameter becomes irrelevant - the single point is always at the start value

Documentation link: The method is located in `/xarray/indexes/range_index.py` starting at line 228.

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
+            # With num=1, return single value at start (matching numpy.linspace behavior)
+            # The endpoint parameter is irrelevant when there's only one point
+            pass  # Keep start and stop as-is, RangeCoordinateTransform will handle it

         transform = RangeCoordinateTransform(
             start, stop, num, coord_name, dim, dtype=dtype
```