# Bug Report: xarray RangeIndex.arange Negative Size

**Target**: `xarray.indexes.range_index.RangeIndex.arange`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `RangeIndex.arange` method creates an index with a negative size when the step direction doesn't match the start-to-stop direction. This violates a fundamental invariant that dimension sizes must be non-negative, and can lead to unexpected behavior or crashes in downstream code.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from xarray.indexes.range_index import RangeIndex

@given(
    start=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    stop=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    step=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
)
@settings(max_examples=1000)
def test_arange_size_nonnegative(start, stop, step):
    assume(step != 0)
    assume(abs(step) > 1e-10)

    index = RangeIndex.arange(start, stop, step, dim="x")
    assert index.size >= 0, f"Size must be non-negative, got {index.size}"
```

**Failing input**: `start=1.0, stop=0.0, step=1.0`

## Reproducing the Bug

```python
from xarray.indexes.range_index import RangeIndex

index = RangeIndex.arange(1.0, 0.0, 1.0, dim="x")
print(f"Size: {index.size}")
```

**Output**:
```
Size: -1
```

**Expected**: Size should be 0 (empty range), similar to `numpy.arange(1.0, 0.0, 1.0)` which returns an empty array.

## Why This Is A Bug

When using a positive step (1.0) to go from 1.0 to 0.0, the range is impossible to traverse in that direction. The correct behavior is to return an empty range (size=0), not a negative size.

The current implementation computes `size = math.ceil((stop - start) / step)`, which gives:
- `(0.0 - 1.0) / 1.0 = -1.0`
- `math.ceil(-1.0) = -1`

This is inconsistent with numpy's behavior:
```python
import numpy as np
np.arange(1.0, 0.0, 1.0)  # array([], dtype=float64)
```

A negative size violates the invariant that array dimensions must be non-negative and can cause crashes in other parts of xarray that expect valid dimension sizes.

## Fix

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -216,7 +216,13 @@ class RangeIndex(CoordinateTransformIndex):
         if coord_name is None:
             coord_name = dim

-        size = math.ceil((stop - start) / step)
+        # Calculate size, ensuring it's non-negative
+        # If step direction doesn't match start-to-stop direction, size should be 0
+        if (stop - start) * step > 0:
+            size = math.ceil((stop - start) / step)
+        else:
+            # Empty range (step goes wrong direction or stop == start)
+            size = 0

         transform = RangeCoordinateTransform(
             start, stop, size, coord_name, dim, dtype=dtype