# Bug Report: xarray.backends.chunks.build_grid_chunks Invalid Chunk Sum

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_grid_chunks` function returns chunks whose sum exceeds the input `size` when `size < chunk_size`, violating the fundamental invariant that chunks must partition the data exactly.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@given(
    size=st.integers(min_value=1, max_value=10000),
    chunk_size=st.integers(min_value=1, max_value=1000),
    region_start=st.integers(min_value=0, max_value=1000) | st.none(),
)
@settings(max_examples=500)
def test_build_grid_chunks_sum_property(size, chunk_size, region_start):
    if region_start is not None:
        region = slice(region_start, region_start + size)
    else:
        region = None

    chunks = build_grid_chunks(size, chunk_size, region)

    assert sum(chunks) == size
    assert all(c > 0 for c in chunks)
```

**Failing input**: `size=1, chunk_size=2, region_start=None`

## Reproducing the Bug

```python
from xarray.backends.chunks import build_grid_chunks

size = 1
chunk_size = 2

chunks = build_grid_chunks(size, chunk_size, region=None)

print(f"Input: size={size}, chunk_size={chunk_size}")
print(f"Output: chunks={chunks}")
print(f"Sum of chunks: {sum(chunks)}")
print(f"Expected sum: {size}")
```

Output:
```
Input: size=1, chunk_size=2
Output: chunks=(2, 1)
Sum of chunks: 3
Expected sum: 1
```

## Why This Is A Bug

The function is supposed to partition a dimension of a given size into chunks aligned with a grid of a specific chunk size. The fundamental invariant is that the sum of returned chunks must equal the input size. When `size < chunk_size`, the function incorrectly returns chunks summing to more than the input size, which would cause data corruption or crashes when used for array chunking.

This can occur in realistic scenarios where a user has a small array (e.g., 1 element) and the backend expects larger chunks (e.g., 2 elements).

## Fix

The bug occurs because the first chunk size is calculated as `chunk_size - (region_start % chunk_size)` without checking if this exceeds the total size. The fix is to cap the first chunk at the remaining size:

```diff
--- a/chunks.py
+++ b/chunks.py
@@ -143,7 +143,7 @@ def build_grid_chunks(

     region_start = region.start or 0
     # Generate the zarr chunks inside the region of this dim
-    chunks_on_region = [chunk_size - (region_start % chunk_size)]
+    chunks_on_region = [min(size, chunk_size - (region_start % chunk_size))]
     chunks_on_region.extend([chunk_size] * ((size - chunks_on_region[0]) // chunk_size))
     if (size - chunks_on_region[0]) % chunk_size != 0:
         chunks_on_region.append((size - chunks_on_region[0]) % chunk_size)
```