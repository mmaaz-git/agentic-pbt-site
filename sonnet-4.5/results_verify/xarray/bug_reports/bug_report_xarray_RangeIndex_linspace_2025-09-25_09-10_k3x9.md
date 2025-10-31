# Bug Report: RangeIndex.linspace Division by Zero

**Target**: `xarray.indexes.RangeIndex.linspace`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`RangeIndex.linspace` crashes with a `ZeroDivisionError` when called with `num=1` and `endpoint=True` due to division by `(num - 1)`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.indexes import RangeIndex


@given(
    start=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    stop=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
)
def test_linspace_num_1_endpoint_true(start, stop):
    """Test that linspace works with num=1 and endpoint=True."""
    idx = RangeIndex.linspace(start, stop, num=1, endpoint=True, dim="x")
    assert idx.size == 1
```

**Failing input**: `start=0.0, stop=0.0` (or any values)

## Reproducing the Bug

```python
from xarray.indexes import RangeIndex

idx = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
```

**Output:**
```
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

The `linspace` method is supposed to create a linearly spaced range similar to `numpy.linspace`. NumPy's `linspace` handles `num=1` correctly by returning a single value at `start` (or `stop` if `endpoint=True`). However, xarray's implementation crashes instead.

The bug occurs at line 283 of `xarray/indexes/range_index.py`:

```python
if endpoint:
    stop += (stop - start) / (num - 1)  # Division by zero when num=1!
```

When `num=1`, the denominator `(num - 1)` becomes 0, causing a division by zero error.

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
+            # When num=1, the single point should be at start
+            stop = start

         transform = RangeCoordinateTransform(
             start, stop, num, coord_name, dim, dtype=dtype
```