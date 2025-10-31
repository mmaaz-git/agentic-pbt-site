# Bug Report: dask.array.overlap _overlap_internal_chunks Type Inconsistency

**Target**: `dask.array.overlap._overlap_internal_chunks`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_overlap_internal_chunks` function returns inconsistent types in its output structure. When processing dimensions with a single chunk, it preserves the original tuple, but for dimensions with multiple chunks, it converts them to lists. This results in a list containing a mix of tuples and lists, violating type consistency expectations.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.array.overlap import _overlap_internal_chunks


@given(
    num_dims=st.integers(min_value=1, max_value=4),
    depth=st.integers(min_value=0, max_value=5)
)
@settings(max_examples=500)
def test_overlap_internal_chunks_inconsistent_types(num_dims, depth):
    chunks = []
    for i in range(num_dims):
        if i % 2 == 0:
            chunks.append(tuple([10]))
        else:
            chunks.append(tuple([5, 5]))

    chunks = tuple(chunks)
    axes = {i: depth for i in range(num_dims)}

    result = _overlap_internal_chunks(chunks, axes)

    types_are_consistent = all(isinstance(r, tuple) for r in result) or \
                          all(isinstance(r, list) for r in result)

    assert types_are_consistent, \
        f"Type inconsistency: result contains {[type(r).__name__ for r in result]}"
```

**Failing input**: `chunks=((10,), (5, 5), (10,))`, `axes={0: 0, 1: 0, 2: 0}`

## Reproducing the Bug

```python
from dask.array.overlap import _overlap_internal_chunks

chunks = ((10,), (5, 5), (10,))
axes = {0: 0, 1: 0, 2: 0}

result = _overlap_internal_chunks(chunks, axes)

print(f"Input:  {chunks}")
print(f"Result: {result}")
print(f"Result element types: {[type(r).__name__ for r in result]}")

assert isinstance(result[0], tuple)
assert isinstance(result[1], list)
assert isinstance(result[2], tuple)
```

## Why This Is A Bug

The function accepts a tuple of tuples (the standard format for dask chunk specifications) but returns a list containing a mix of tuples and lists. Specifically:

- Dimensions with a single chunk: returned as tuples (line 39: `chunks.append(bds)`)
- Dimensions with multiple chunks: returned as lists (line 46: `chunks.append(left + mid + right)` where left, mid, right are lists)

This type inconsistency violates the principle of uniform data structures and could cause subtle bugs if downstream code expects consistent types. While the Array constructor likely normalizes this, the inconsistency is still a contract violation.

## Fix

```diff
--- a/dask/array/overlap.py
+++ b/dask/array/overlap.py
@@ -36,7 +36,7 @@ def _overlap_internal_chunks(original_chunks, axes):
             right_depth = depth

         if len(bds) == 1:
-            chunks.append(bds)
+            chunks.append(list(bds))
         else:
             left = [bds[0] + right_depth]
             right = [bds[-1] + left_depth]
```

Alternatively, convert all results to tuples for consistency:

```diff
--- a/dask/array/overlap.py
+++ b/dask/array/overlap.py
@@ -44,7 +44,7 @@ def _overlap_internal_chunks(original_chunks, axes):
             for bd in bds[1:-1]:
                 mid.append(bd + left_depth + right_depth)
-            chunks.append(left + mid + right)
+            chunks.append(tuple(left + mid + right))
     return chunks
```