# Bug Report: xarray RangeIndex.linspace Division by Zero

**Target**: `xarray.indexes.range_index.RangeIndex.linspace`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `RangeIndex.linspace` method crashes with a `ZeroDivisionError` when called with `num=1` and `endpoint=True`. This occurs because the method attempts to compute `(stop - start) / (num - 1)` without checking if `num == 1`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
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
```

**Failing input**: `start=0.0, stop=1.0, num=1, endpoint=True`

## Reproducing the Bug

```python
from xarray.indexes.range_index import RangeIndex

index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
```

**Error**:
```
ZeroDivisionError: float division by zero
  File "xarray/indexes/range_index.py", line 283, in linspace
    stop += (stop - start) / (num - 1)
```

## Why This Is A Bug

When `num=1`, the expression `(num - 1)` evaluates to 0, causing a division by zero error. This is a valid use case - a user might want to create a single-point range index. The function should handle this edge case gracefully, similar to how `numpy.linspace` does:

```python
import numpy as np
np.linspace(0.0, 1.0, num=1, endpoint=True)  # Works: array([0.])
```

## Fix

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -280,7 +280,10 @@ class RangeIndex(CoordinateTransformIndex):
         if coord_name is None:
             coord_name = dim

-        if endpoint:
+        if endpoint and num == 1:
+            # Special case: single point, use start as the value
+            pass  # stop remains unchanged
+        elif endpoint:
             stop += (stop - start) / (num - 1)

         transform = RangeCoordinateTransform(