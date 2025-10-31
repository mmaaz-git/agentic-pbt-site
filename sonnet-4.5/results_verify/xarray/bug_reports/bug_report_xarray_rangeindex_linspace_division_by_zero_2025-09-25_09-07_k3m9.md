# Bug Report: xarray RangeIndex.linspace Division by Zero

**Target**: `xarray.indexes.RangeIndex.linspace`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`RangeIndex.linspace` crashes with a division by zero error when called with `num=1` and `endpoint=True`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from xarray.indexes import RangeIndex


@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    num=st.integers(min_value=1, max_value=1000),
)
@settings(max_examples=200)
def test_linspace_with_endpoint(start, stop, num):
    assume(start != stop)

    index = RangeIndex.linspace(
        start=start,
        stop=stop,
        num=num,
        endpoint=True,
        dim="x"
    )
    assert index.size == num
```

**Failing input**: `start=0.0, stop=1.0, num=1`

## Reproducing the Bug

```python
from xarray.indexes import RangeIndex

index = RangeIndex.linspace(
    start=0.0,
    stop=1.0,
    num=1,
    endpoint=True,
    dim="x"
)
```

Output:
```
ZeroDivisionError: float division by zero
```

The error occurs at `range_index.py:283`:
```python
if endpoint:
    stop += (stop - start) / (num - 1)
```

When `num=1`, this divides by zero.

## Why This Is A Bug

The documentation for `RangeIndex.linspace` does not specify that `num` must be greater than 1 when `endpoint=True`. Users would reasonably expect to create a single-point index at the start position. NumPy's `linspace` handles this case correctly by returning an array with just the start value.

## Fix

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -280,7 +280,10 @@ class RangeIndex(CoordinateTransformIndex):
         if coord_name is None:
             coord_name = dim

         if endpoint:
-            stop += (stop - start) / (num - 1)
+            if num == 1:
+                stop = start
+            else:
+                stop += (stop - start) / (num - 1)

         transform = RangeCoordinateTransform(
             start, stop, num, coord_name, dim, dtype=dtype
```