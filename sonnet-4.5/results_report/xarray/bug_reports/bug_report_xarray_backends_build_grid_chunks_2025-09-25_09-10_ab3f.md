# Bug Report: xarray.backends.chunks.build_grid_chunks Invalid Chunk Sum

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_grid_chunks` function returns chunks whose sum does not equal the input `size` when the first chunk would be larger than the total size.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.backends.chunks import build_grid_chunks

@given(
    size=st.integers(min_value=1, max_value=1000),
    chunk_size=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=500)
def test_build_grid_chunks_sum_invariant(size, chunk_size):
    chunks = build_grid_chunks(size, chunk_size)
    assert sum(chunks) == size, f"Sum of chunks {sum(chunks)} != size {size}"
```

**Failing input**: `size=1, chunk_size=2`

## Reproducing the Bug

```python
from xarray.backends.chunks import build_grid_chunks

size = 1
chunk_size = 2

chunks = build_grid_chunks(size, chunk_size)
print(f"size={size}, chunk_size={chunk_size}")
print(f"chunks={chunks}")
print(f"sum(chunks)={sum(chunks)}")
```

Output:
```
size=1, chunk_size=2
chunks=(2, 1)
sum(chunks)=3
```

The sum of chunks (3) does not equal the size (1).

## Why This Is A Bug

The function's fundamental invariant is that the returned chunks should partition the input size. The docstring doesn't explicitly state this, but it's implied by the function's purpose (building a grid of chunks that cover a dimension). When `sum(chunks) != size`, the chunks either cover too much or too little of the intended range, which would cause data corruption or access errors.

The bug occurs because:
1. Line 146 computes the first chunk as `chunk_size - (region_start % chunk_size)`, which can exceed `size`
2. Line 147 then computes remaining chunks based on `size - chunks_on_region[0]`, which becomes negative
3. Line 148-149 incorrectly handle the negative remainder

## Fix

```diff
--- a/xarray/backends/chunks.py
+++ b/xarray/backends/chunks.py
@@ -143,7 +143,8 @@ def build_grid_chunks(

     region_start = region.start or 0
     # Generate the zarr chunks inside the region of this dim
-    chunks_on_region = [chunk_size - (region_start % chunk_size)]
+    first_chunk_size = min(size, chunk_size - (region_start % chunk_size))
+    chunks_on_region = [first_chunk_size]
     chunks_on_region.extend([chunk_size] * ((size - chunks_on_region[0]) // chunk_size))
     if (size - chunks_on_region[0]) % chunk_size != 0:
         chunks_on_region.append((size - chunks_on_region[0]) % chunk_size)
```