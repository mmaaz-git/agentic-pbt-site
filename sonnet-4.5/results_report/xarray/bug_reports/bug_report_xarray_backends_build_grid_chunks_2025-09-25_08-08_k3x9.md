# Bug Report: xarray.backends.chunks.build_grid_chunks Sum Invariant Violation

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_grid_chunks` function violates a fundamental invariant: the sum of returned chunks does not equal the input `size` parameter when `size < chunk_size`. This causes the function to return chunks that represent more data than exists.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.backends.chunks import build_grid_chunks

@given(
    size=st.integers(min_value=1, max_value=10000),
    chunk_size=st.integers(min_value=1, max_value=1000)
)
@settings(max_examples=1000)
def test_sum_equals_size(size, chunk_size):
    chunks = build_grid_chunks(size, chunk_size, region=None)
    assert sum(chunks) == size
```

**Failing input**: `size=1, chunk_size=2`

## Reproducing the Bug

```python
from xarray.backends.chunks import build_grid_chunks

result = build_grid_chunks(size=1, chunk_size=2, region=None)

print(f"Chunks: {result}")
print(f"Sum of chunks: {sum(result)}")
print(f"Expected size: 1")
```

Output:
```
Chunks: (2, 1)
Sum of chunks: 3
Expected size: 1
```

## Why This Is A Bug

The function's purpose is to divide a dimension of given `size` into chunks of `chunk_size`. The fundamental invariant is that the sum of all chunk sizes must equal the total size. When `size < chunk_size`, the function incorrectly creates:
1. A first chunk of size `chunk_size` (larger than the total size)
2. An additional chunk containing the "remainder"

This results in `sum(chunks) > size`, violating the mathematical property that chunks should partition the dimension.

This bug would affect real users when working with small arrays or array dimensions smaller than the configured chunk size, which is a common scenario in data processing pipelines.

## Fix

```diff
--- a/xarray/backends/chunks.py
+++ b/xarray/backends/chunks.py
@@ -143,7 +143,7 @@ def build_grid_chunks(

     region_start = region.start or 0
     # Generate the zarr chunks inside the region of this dim
-    chunks_on_region = [chunk_size - (region_start % chunk_size)]
+    chunks_on_region = [min(chunk_size - (region_start % chunk_size), size)]
     chunks_on_region.extend([chunk_size] * ((size - chunks_on_region[0]) // chunk_size))
     if (size - chunks_on_region[0]) % chunk_size != 0:
         chunks_on_region.append((size - chunks_on_region[0]) % chunk_size)
```