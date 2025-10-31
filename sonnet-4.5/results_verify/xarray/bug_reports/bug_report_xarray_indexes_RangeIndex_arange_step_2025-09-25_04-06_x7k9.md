# Bug Report: RangeIndex.arange Step Parameter Not Preserved

**Target**: `xarray.indexes.RangeIndex.arange`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`RangeIndex.arange` does not preserve the `step` parameter when creating the range. The actual step used in the resulting index is different from the requested step, violating the documented API contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from xarray.indexes import RangeIndex
import numpy as np

@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    step=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=500)
def test_arange_step_matches_parameter(start, stop, step):
    assume(stop > start)
    assume(step > 0)
    assume((stop - start) / step < 1e6)

    index = RangeIndex.arange(start, stop, step, dim="x")

    coords = index.transform.forward({index.dim: np.arange(index.size)})
    values = coords[index.coord_name]

    if index.size > 1:
        actual_steps = np.diff(values)

        assert np.allclose(actual_steps, step, rtol=1e-9)
```

**Failing input**: `start=0.0, stop=1.5, step=1.0`

## Reproducing the Bug

```python
from xarray.indexes import RangeIndex
import numpy as np

index = RangeIndex.arange(0.0, 1.5, 1.0, dim="x")

print(f"Expected step: 1.0")
print(f"Actual step: {index.step}")

coords = index.transform.forward({index.dim: np.arange(index.size)})
values = coords[index.coord_name]

print(f"Expected values: [0.0, 1.0]")
print(f"Actual values: {values}")
```

Output:
```
Expected step: 1.0
Actual step: 0.75
Expected values: [0.0, 1.0]
Actual values: [0.   0.75]
```

## Why This Is A Bug

The documentation at line 155 states: "with spacing between values given by step". The example in the docstring (lines 190-201) also demonstrates that the step parameter should determine the spacing between values.

The root cause is at lines 219-223:
1. The code calculates `size = math.ceil((stop - start) / step)` = `math.ceil(1.5 / 1.0)` = 2
2. It then creates `RangeCoordinateTransform(start, stop, size, ...)`
3. The `step` property of `RangeCoordinateTransform` recalculates step as `(stop - start) / size` = `1.5 / 2` = 0.75

This means the `step` parameter is only used to compute the size, but the actual step is recomputed and can differ significantly.

## Fix

The fix is to store the step parameter and use it in the transform, rather than recalculating it:

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -219,7 +219,8 @@ class RangeIndex(CoordinateTransformIndex):
         size = math.ceil((stop - start) / step)

         transform = RangeCoordinateTransform(
-            start, stop, size, coord_name, dim, dtype=dtype
+            start, start + size * step, size, coord_name, dim, dtype=dtype
         )
+        transform._step = step

         return cls(transform)
```