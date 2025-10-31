# Bug Report: dask.array.overlap periodic/reflect Crash with Large Depth

**Target**: `dask.array.overlap.periodic` and `dask.array.overlap.reflect`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `periodic` and `reflect` functions crash with a `ValueError` when `depth` is greater than the array size along the specified axis, while NumPy's `np.pad` handles this case correctly.

## Property-Based Test

```python
import numpy as np
import dask.array as da
from dask.array.overlap import periodic, reflect
from hypothesis import given, strategies as st, assume, settings

@given(
    arr_size=st.integers(min_value=5, max_value=30),
    axis=st.integers(min_value=0, max_value=1),
    depth=st.integers(min_value=1, max_value=10),
    chunk_size=st.integers(min_value=3, max_value=10)
)
@settings(max_examples=200)
def test_periodic_size_increase(arr_size, axis, depth, chunk_size):
    shape = (arr_size, arr_size)
    arr = da.arange(np.prod(shape)).reshape(shape).rechunk(chunk_size)

    result = periodic(arr, axis, depth)

    expected_shape = list(shape)
    expected_shape[axis] += 2 * depth
    assert result.shape == tuple(expected_shape)
```

**Failing input**: `arr_size=5, axis=0, depth=6, chunk_size=3`

## Reproducing the Bug

```python
import dask.array as da
from dask.array.overlap import periodic

arr = da.arange(25).reshape(5, 5).rechunk(3)
result = periodic(arr, axis=0, depth=6)
```

Running this code produces:
```
ValueError: Chunks do not add up to shape. Got chunks=((6,), (3, 2)), shape=(5, 5)
```

The same issue occurs with the `reflect` function.

## Why This Is A Bug

1. **No documented limitation**: The function documentation does not specify that `depth` must be less than the array size.

2. **Inconsistent with NumPy**: NumPy's `np.pad` function handles depth values larger than the array size:
   ```python
   import numpy as np
   np_arr = np.arange(5)
   result = np.pad(np_arr, (6, 6), mode='wrap')
   # Works fine, returns array of shape (17,)
   ```

3. **Confusing error message**: The error message "Chunks do not add up to shape" is an internal implementation detail that doesn't help users understand the actual limitation.

4. **Valid use case**: Users might legitimately want to apply periodic or reflective boundaries with depths larger than the array size, especially for small arrays or large boundary regions.

## Fix

The bug occurs in the `_remove_overlap_boundaries` function at line 276-283 of `overlap.py`. The function tries to rechunk the boundary slices to have size `depth`, but when `depth > arr_size`, the slice itself is smaller than `depth`.

```python
def _remove_overlap_boundaries(l, r, axis, depth):
    lchunks = list(l.chunks)
    lchunks[axis] = (depth,)
    rchunks = list(r.chunks)
    rchunks[axis] = (depth,)

    l = l.rechunk(tuple(lchunks))  # Fails when l.shape[axis] < depth
    r = r.rechunk(tuple(rchunks))
    return l, r
```

The fix should handle the case where the extracted boundary is smaller than the requested depth by either:
1. Not rechunking when the slice is already smaller than depth
2. Or using the actual slice size instead of forcing it to be `depth`

```diff
--- a/overlap.py
+++ b/overlap.py
@@ -276,10 +276,14 @@ def _remove_overlap_boundaries(l, r, axis, depth):
 def _remove_overlap_boundaries(l, r, axis, depth):
     lchunks = list(l.chunks)
-    lchunks[axis] = (depth,)
+    actual_l_size = sum(l.chunks[axis])
+    lchunks[axis] = (min(depth, actual_l_size),)
     rchunks = list(r.chunks)
-    rchunks[axis] = (depth,)
+    actual_r_size = sum(r.chunks[axis])
+    rchunks[axis] = (min(depth, actual_r_size),)

-    l = l.rechunk(tuple(lchunks))
-    r = r.rechunk(tuple(rchunks))
+    if lchunks != list(l.chunks):
+        l = l.rechunk(tuple(lchunks))
+    if rchunks != list(r.chunks):
+        r = r.rechunk(tuple(rchunks))
     return l, r
```