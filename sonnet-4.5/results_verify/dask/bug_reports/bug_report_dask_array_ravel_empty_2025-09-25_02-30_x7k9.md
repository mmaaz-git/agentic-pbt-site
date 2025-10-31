# Bug Report: dask.array ravel() crashes on empty arrays with non-zero dimensions

**Target**: `dask.array.ravel()` and `dask.array.reshape()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`dask.array.ravel()` crashes with `TypeError: reduce() of empty iterable with no initial value` when called on empty arrays that have non-zero dimensions (e.g., shape `(3, 0)`). NumPy handles these cases correctly, but Dask fails due to missing initial value in `reduce()` call in the reshape logic.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays, array_shapes
import dask.array as da
import numpy as np


@given(
    arr=arrays(
        dtype=np.float64,
        shape=array_shapes(min_dims=1, max_dims=2, min_side=0, max_side=10)
    )
)
@settings(max_examples=100, deadline=None)
def test_ravel_preserves_elements(arr):
    darr = da.from_array(arr, chunks=2)
    raveled = da.ravel(darr)
    assert raveled.ndim == 1
    assert raveled.size == arr.size
    result = raveled.compute()
    expected = np.ravel(arr)
    np.testing.assert_array_equal(result, expected)
```

**Failing input**: `arr=np.empty((3, 0), dtype=np.float64)`

## Reproducing the Bug

```python
import dask.array as da
import numpy as np

arr = np.empty((3, 0), dtype=np.float64)
darr = da.from_array(arr, chunks=2)
raveled = da.ravel(darr)
```

Output:
```
TypeError: reduce() of empty iterable with no initial value
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/reshape.py", line 105, in reshape_rechunk
    if reduce(mul, outshape[oleft : oi + 1]) != din:
```

## Why This Is A Bug

Empty arrays with non-zero dimensions (like `(3, 0)`) are valid NumPy arrays that should be supported by Dask. NumPy's `ravel()` correctly handles these arrays and returns a 1D empty array `(0,)`. Dask should provide the same behavior for API compatibility.

The bug occurs because `reshape_rechunk()` uses `reduce(mul, outshape[oleft : oi + 1])` without providing an initial value. When the slice is empty (as happens with empty arrays), `reduce()` raises `TypeError`.

## Fix

The fix is to provide an initial value of `1` to the `reduce()` calls in `reshape_rechunk()`, since 1 is the multiplicative identity.

```diff
--- a/dask/array/reshape.py
+++ b/dask/array/reshape.py
@@ -102,7 +102,7 @@ def reshape_rechunk(inshape, outshape, inchunks, disallow_dimension_expansion=F
                 oleft = oi - 1
                 while oleft >= 0 and reduce(mul, outshape[oleft : oi + 1]) < din:
                     oleft -= 1
-                if reduce(mul, outshape[oleft : oi + 1]) != din:
+                if reduce(mul, outshape[oleft : oi + 1], 1) != din:
                     raise NotImplementedError(_not_implemented_message)
                 oleft -= 1
                 # Expand input chunks to match output dimension
```

A similar fix may be needed for line 79 where `reduce(mul, inshape[ileft : ii + 1])` is called:

```diff
@@ -76,7 +76,7 @@ def reshape_rechunk(inshape, outshape, inchunks, disallow_dimension_expansion=F
                     mapper_in[ileft] = oi
                     ileft -= 1

-                if reduce(mul, inshape[ileft : ii + 1]) != dout:
+                if reduce(mul, inshape[ileft : ii + 1], 1) != dout:
                     raise NotImplementedError(_not_implemented_message)
                 # Special case to avoid intermediate rechunking:
                 # When all the lower axis are completely chunked (chunksize=1) then
```