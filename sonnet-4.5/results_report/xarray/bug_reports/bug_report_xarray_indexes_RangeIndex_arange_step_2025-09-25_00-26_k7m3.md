# Bug Report: xarray.indexes.RangeIndex.arange Incorrect Step Calculation

**Target**: `xarray.indexes.RangeIndex.arange`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`RangeIndex.arange()` incorrectly recalculates the step size based on the range size instead of using the provided step parameter, leading to incorrect coordinate values that don't match `numpy.arange()` semantics.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import assume, given, strategies as st
from xarray.indexes import RangeIndex


@given(
    start=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    step=st.floats(min_value=0.01, max_value=10, allow_nan=False, allow_infinity=False),
)
def test_arange_matches_numpy_semantics(start, stop, step):
    assume(stop > start)

    idx = RangeIndex.arange(start, stop, step, dim="x")
    np_range = np.arange(start, stop, step)

    assert idx.size == len(np_range)

    xr_values = idx.transform.forward({"x": np.arange(idx.size)})["x"]
    assert np.allclose(xr_values, np_range)
```

**Failing input**: `start=-1.5, stop=0.0, step=1.0`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from xarray.indexes import RangeIndex

start = -1.5
stop = 0.0
step = 1.0

idx = RangeIndex.arange(start, stop, step, dim="x")
xr_values = idx.transform.forward({"x": np.arange(idx.size)})["x"]
np_values = np.arange(start, stop, step)

print(f"XArray values: {xr_values}")
print(f"NumPy values:  {np_values}")
print(f"XArray step: {idx.step}")
print(f"Expected step: {step}")
```

Output:
```
XArray values: [-1.5  -0.75]
NumPy values:  [-1.5 -0.5 ]
XArray step: 0.75
Expected step: 1.0
```

## Why This Is A Bug

The `RangeIndex.arange()` method is documented to be similar to `numpy.arange()` (line 120 docstring), but it produces different values. The root cause is:

1. Line 219 calculates `size = math.ceil((stop - start) / step)` = ceil(1.5 / 1.0) = 2
2. Line 221-223 creates `RangeCoordinateTransform(start, stop, size, ...)` without passing the step
3. Line 62 recalculates step as `(self.stop - self.start) / self.size` = (0.0 - (-1.5)) / 2 = 0.75

This recalculated step (0.75) differs from the input step (1.0), causing incorrect coordinate values.

## Fix

The fix is to pass the step parameter to `RangeCoordinateTransform` and store it explicitly. Here's the patch:

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -27,12 +27,13 @@ class RangeCoordinateTransform(CoordinateTransform):
     def __init__(
         self,
         start: float,
         stop: float,
         size: int,
         coord_name: Hashable,
         dim: str,
         dtype: Any = None,
+        step: float | None = None,
     ):
         if dtype is None:
             dtype = np.dtype(np.float64)
@@ -40,7 +41,7 @@ class RangeCoordinateTransform(CoordinateTransform):
         super().__init__([coord_name], {dim: size}, dtype=dtype)

         self.start = start
         self.stop = stop
-        self._step = None  # Will be calculated by property
+        self._step = step

     @property
     def coord_name(self) -> Hashable:
@@ -216,7 +217,7 @@ class RangeIndex(CoordinateTransformIndex):
         if coord_name is None:
             coord_name = dim

         size = math.ceil((stop - start) / step)

         transform = RangeCoordinateTransform(
-            start, stop, size, coord_name, dim, dtype=dtype
+            start, stop, size, coord_name, dim, dtype=dtype, step=step
         )

         return cls(transform)
```