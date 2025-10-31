# Bug Report: RangeIndex.linspace Division by Zero

**Target**: `xarray.indexes.RangeIndex.linspace`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`RangeIndex.linspace` crashes with `ZeroDivisionError` when called with `num=1` and `endpoint=True`.

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
```

**Failing input**: `start=0.0, stop=1.0, num=1, endpoint=True`

## Reproducing the Bug

```python
from xarray.indexes import RangeIndex

index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
```

## Why This Is A Bug

The `linspace` method is supposed to create a range with `num` evenly spaced points. When `num=1`, there is only one point, so it should return either `start` or `stop` depending on whether `endpoint=True`. However, the implementation at line 283 computes `stop += (stop - start) / (num - 1)`, which causes division by zero when `num=1`.

This is analogous to `numpy.linspace(0, 1, 1)` which returns `array([0.])` without error.

## Fix

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -280,7 +280,10 @@ class RangeIndex(CoordinateTransformIndex):
         if coord_name is None:
             coord_name = dim

-        if endpoint:
+        if endpoint and num > 1:
             stop += (stop - start) / (num - 1)
+        elif endpoint and num == 1:
+            # With num=1 and endpoint=True, the single point should be at stop
+            start = stop

         transform = RangeCoordinateTransform(
             start, stop, num, coord_name, dim, dtype=dtype
```