# Bug Report: dask.array.overlap.ensure_minimum_chunksize Incorrect Parameter Documentation

**Target**: `dask.array.overlap.ensure_minimum_chunksize`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstring for `ensure_minimum_chunksize` incorrectly describes the `size` parameter as "The maximum size of any chunk" when it should be "The minimum size of any chunk".

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.array.overlap import ensure_minimum_chunksize


@settings(max_examples=200)
@given(
    chunks=st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=10),
    size=st.integers(min_value=1, max_value=30)
)
def test_ensure_minimum_chunksize_property_all_chunks_at_least_size(chunks, size):
    chunks = tuple(chunks)
    if sum(chunks) < size:
        return

    try:
        result = ensure_minimum_chunksize(size, chunks)

        for chunk in result:
            assert chunk >= size, f"Chunk {chunk} is less than minimum size {size}"
    except ValueError as e:
        if "larger than your array" not in str(e):
            raise
```

**Failing input**: N/A - this is a documentation bug

## Reproducing the Bug

The docstring says `size` is "The maximum size of any chunk", but the function ensures all chunks are >= size (i.e., it's the minimum).

```python
from dask.array.overlap import ensure_minimum_chunksize

result = ensure_minimum_chunksize(10, (20, 20, 1))
print(result)

print(f"All chunks >= 10: {all(c >= 10 for c in result)}")
```

Output:
```
(20, 11, 10)
All chunks >= 10: True
```

## Why This Is A Bug

The function is named `ensure_MINIMUM_chunksize`, the docstring summary says "Determine new chunks to ensure that every chunk >= size", and the implementation ensures all output chunks are at least `size`. However, the parameter documentation incorrectly states:

```
size: int
    The maximum size of any chunk.
```

This contradicts the function name, summary, examples, and implementation.

## Fix

```diff
--- a/dask/array/overlap.py
+++ b/dask/array/overlap.py
@@ -509,7 +509,7 @@ def ensure_minimum_chunksize(size, chunks):

     Parameters
     ----------
     size: int
-        The maximum size of any chunk.
+        The minimum size of any chunk.
     chunks: tuple
         Chunks along one axis, e.g. ``(3, 3, 2)``
```