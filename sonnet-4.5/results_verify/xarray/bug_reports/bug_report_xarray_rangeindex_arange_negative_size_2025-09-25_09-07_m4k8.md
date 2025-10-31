# Bug Report: xarray RangeIndex.arange Negative Size

**Target**: `xarray.indexes.RangeIndex.arange`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`RangeIndex.arange` creates an index with negative size when `start`, `stop`, and `step` have incompatible signs.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from xarray.indexes import RangeIndex


@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    step=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_arange_step_nonzero(start, stop, step):
    assume(step != 0)
    assume(abs(step) > 1e-10)
    assume((stop - start) / step < 1e6)
    assume((stop - start) / step > -1e6)

    index = RangeIndex.arange(
        start=start,
        stop=stop,
        step=step,
        dim="x"
    )
    assert index.size >= 0
```

**Failing input**: `start=1.0, stop=0.0, step=1.0`

## Reproducing the Bug

```python
import math
import numpy as np
from xarray.indexes import RangeIndex

index = RangeIndex.arange(start=1.0, stop=0.0, step=1.0, dim="x")
print(f"Size: {index.size}")
print(f"Start: {index.start}, Stop: {index.stop}, Step: {index.step}")

size_computed = math.ceil((0.0 - 1.0) / 1.0)
print(f"\nSize computed as: math.ceil((stop - start) / step) = math.ceil(-1.0) = {size_computed}")

print(f"\nNumPy's behavior:")
arr = np.arange(1.0, 0.0, 1.0)
print(f"np.arange(1.0, 0.0, 1.0) = {arr}")
print(f"np.arange(1.0, 0.0, 1.0).size = {arr.size}")
```

Output:
```
Size: -1
Start: 1.0, Stop: 0.0, Step: 1.0

Size computed as: math.ceil((stop - start) / step) = math.ceil(-1.0) = -1

NumPy's behavior:
np.arange(1.0, 0.0, 1.0) = []
np.arange(1.0, 0.0, 1.0).size = 0
```

The bug occurs at `range_index.py:219`:
```python
size = math.ceil((stop - start) / step)
```

When the step direction doesn't match the start-to-stop direction, this produces a negative size.

## Why This Is A Bug

A dimension size should never be negative. NumPy's `arange` handles this by returning an empty array (size 0) when the step direction is incompatible with start and stop. xarray's `RangeIndex.arange` should either do the same or raise a clear error message.

## Fix

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -217,7 +217,7 @@ class RangeIndex(CoordinateTransformIndex):
         if coord_name is None:
             coord_name = dim

-        size = math.ceil((stop - start) / step)
+        size = max(0, math.ceil((stop - start) / step))

         transform = RangeCoordinateTransform(
             start, stop, size, coord_name, dim, dtype=dtype
```