# Bug Report: RangeIndex.linspace Division by Zero

**Target**: `xarray.indexes.RangeIndex.linspace`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`RangeIndex.linspace` raises a `ZeroDivisionError` when called with `num=1` and `endpoint=True` due to division by `(num - 1)` without checking if `num == 1`.

## Property-Based Test

```python
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
```

**Failing input**: `start=0.0, stop=1.0, num=1, endpoint=True`

## Reproducing the Bug

```python
from xarray.indexes import RangeIndex

index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
```

**Output:**
```
ZeroDivisionError: float division by zero
  File "xarray/indexes/range_index.py", line 283, in linspace
    stop += (stop - start) / (num - 1)
```

## Why This Is A Bug

The `linspace` method is documented as creating a RangeIndex similar to `numpy.linspace`. NumPy's `linspace` correctly handles `num=1` by returning an array with a single element at the `start` position. However, xarray's implementation crashes when `endpoint=True` and `num=1` because it attempts to divide by zero: `(stop - start) / (num - 1)`.

This is a valid use case - users should be able to create a single-point coordinate. The function should handle this edge case gracefully.

## Fix

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
+            # For a single point with endpoint=True, stop should equal start
+            stop = start

         transform = RangeCoordinateTransform(
             start, stop, num, coord_name, dim, dtype=dtype
```