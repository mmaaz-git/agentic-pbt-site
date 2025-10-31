# Bug Report: dask.array.overlap periodic/reflect ValueError When Depth Exceeds Array Size

**Target**: `dask.array.overlap.periodic` and `dask.array.overlap.reflect`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `periodic` and `reflect` functions in dask.array.overlap crash with a `ValueError` when the `depth` parameter is greater than the array size along the specified axis, while NumPy's equivalent `np.pad` handles this case without error.

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

<details>

<summary>
**Failing input**: `arr_size=5, axis=0, depth=6, chunk_size=3`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 25, in <module>
    test_periodic_size_increase()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 7, in test_periodic_size_increase
    arr_size=st.integers(min_value=5, max_value=30),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 17, in test_periodic_size_increase
    result = periodic(arr, axis, depth)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/overlap.py", line 200, in periodic
    l, r = _remove_overlap_boundaries(l, r, axis, depth)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/overlap.py", line 282, in _remove_overlap_boundaries
    l = l.rechunk(tuple(lchunks))
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/core.py", line 2822, in rechunk
    return rechunk(self, chunks, threshold, block_size_limit, balance, method)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/rechunk.py", line 349, in rechunk
    chunks = normalize_chunks(
        chunks, x.shape, limit=block_size_limit, dtype=x.dtype, previous_chunks=x.chunks
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/core.py", line 3225, in normalize_chunks
    raise ValueError(
    ...<2 lines>...
    )
ValueError: Chunks do not add up to shape. Got chunks=((6,), (3, 2)), shape=(5, 5)
Falsifying example: test_periodic_size_increase(
    arr_size=5,
    axis=0,
    depth=6,
    chunk_size=3,
)
```
</details>

## Reproducing the Bug

```python
import dask.array as da
from dask.array.overlap import periodic

# Create a 5x5 array with chunks of size 3
arr = da.arange(25).reshape(5, 5).rechunk(3)

# Try to apply periodic boundaries with depth=6 (larger than array size of 5)
try:
    result = periodic(arr, axis=0, depth=6)
    print(f"Success: result shape is {result.shape}")
except ValueError as e:
    print(f"ValueError: {e}")
```

<details>

<summary>
ValueError: Chunks do not add up to shape
</summary>
```
ValueError: Chunks do not add up to shape. Got chunks=((6,), (3, 2)), shape=(5, 5)
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **NumPy Compatibility Violation**: NumPy's `np.pad` function handles depth values larger than the array size without errors. Both `mode='wrap'` (periodic) and `mode='reflect'` work correctly:
   - `np.pad(np.arange(5), (6, 6), mode='wrap')` produces an array of shape (17,)
   - `np.pad(np.arange(5), (6, 6), mode='reflect')` produces an array of shape (17,)

2. **Undocumented Limitation**: Neither the `periodic()` nor `reflect()` function documentation mentions any restriction on the depth parameter relative to array size. The docstrings are minimal and don't warn users about this constraint.

3. **Inconsistent with Module Philosophy**: The overlap module documentation for `overlap()` and `map_overlap()` explicitly states: "If depth is larger than any chunk along a particular axis, then the array is rechunked." This implies that large depth values should be handled automatically through rechunking, not by raising errors.

4. **Misleading Error Message**: The error "Chunks do not add up to shape" exposes internal implementation details about chunking rather than providing a clear, user-facing message about the actual constraint (depth exceeding array size).

5. **Valid Use Cases Exist**: There are legitimate scenarios where users might need depth > array_size:
   - Testing with small arrays but large boundary regions
   - Algorithms that require multiple wrapping/reflection cycles
   - Porting code from NumPy that already uses such patterns

## Relevant Context

The bug originates in the `_remove_overlap_boundaries()` helper function at lines 276-284 of `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/array/overlap.py`. This function attempts to rechunk boundary slices to have exactly `depth` elements, but fails when the sliced region itself has fewer than `depth` elements.

The function slices the left and right boundaries of the array:
- For `periodic`: takes first `depth` elements and last `depth` elements
- For `reflect`: takes reflected versions of these boundaries

When `depth=6` but the array dimension is only size 5, the slices are at most size 5, making it impossible to rechunk them to size 6.

Related functions like `ensure_minimum_chunksize()` (line 320) already handle similar size constraints elsewhere in the module, suggesting this edge case was overlooked rather than intentionally excluded.

Documentation references:
- overlap.py source: https://github.com/dask/dask/blob/main/dask/array/overlap.py
- NumPy pad documentation: https://numpy.org/doc/stable/reference/generated/numpy.pad.html

## Proposed Fix

The fix modifies `_remove_overlap_boundaries()` to handle cases where the boundary slice is smaller than the requested depth:

```diff
--- a/overlap.py
+++ b/overlap.py
@@ -275,11 +275,19 @@

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
+    # Only rechunk if the chunks actually need to change
+    if lchunks != list(l.chunks):
+        l = l.rechunk(tuple(lchunks))
+    if rchunks != list(r.chunks):
+        r = r.rechunk(tuple(rchunks))
     return l, r
```

This ensures the rechunking operation uses the actual size of the boundary slice rather than forcing it to be `depth`, preventing the ValueError while maintaining correct behavior for cases where depth <= array_size.