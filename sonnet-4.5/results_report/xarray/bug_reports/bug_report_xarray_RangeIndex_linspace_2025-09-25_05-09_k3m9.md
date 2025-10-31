# Bug Report: RangeIndex.linspace Division by Zero

**Target**: `xarray.indexes.RangeIndex.linspace`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`RangeIndex.linspace()` crashes with `ZeroDivisionError` when called with `num=1` and `endpoint=True`, which are valid input values.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from xarray.indexes import RangeIndex

@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    num=st.integers(min_value=1, max_value=1000),
    endpoint=st.booleans()
)
def test_linspace_no_crash(start, stop, num, endpoint):
    try:
        index = RangeIndex.linspace(start, stop, num, endpoint=endpoint, dim="x")
        assert index.size == num
    except ZeroDivisionError:
        assert False, "RangeIndex.linspace crashed with valid inputs"
```

**Failing input**: `RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.indexes import RangeIndex

index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
```

Output:
```
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

The documentation for `RangeIndex.linspace()` does not specify that `num` must be greater than 1. According to the numpy.linspace API (which this method mimics), `num=1` is a valid input. When `num=1` and `endpoint=True`, the code attempts to compute `stop += (stop - start) / (num - 1)`, which results in division by zero.

## Fix

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -280,7 +280,8 @@ class RangeIndex(CoordinateTransformIndex):
             coord_name = dim

         if endpoint:
-            stop += (stop - start) / (num - 1)
+            if num > 1:
+                stop += (stop - start) / (num - 1)

         transform = RangeCoordinateTransform(
             start, stop, num, coord_name, dim, dtype=dtype
```