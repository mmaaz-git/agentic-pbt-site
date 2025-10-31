# Bug Report: xarray.backends.chunks.build_grid_chunks Invalid Sum

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_grid_chunks` function produces chunks that do not sum to the specified `size` parameter when `chunk_size > size`, leading to incorrect chunk calculations that could corrupt data in zarr backends.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from xarray.backends.chunks import build_grid_chunks

@settings(max_examples=500)
@given(
    size=st.integers(min_value=1, max_value=10000),
    chunk_size=st.integers(min_value=1, max_value=1000)
)
def test_build_grid_chunks_sum_invariant(size, chunk_size):
    chunks = build_grid_chunks(size=size, chunk_size=chunk_size, region=None)
    assert sum(chunks) == size
```

**Failing input**: `size=1, chunk_size=2`

## Reproducing the Bug

```python
from xarray.backends.chunks import build_grid_chunks

result = build_grid_chunks(size=1, chunk_size=2, region=None)
print(f"Result: {result}")
print(f"Sum: {sum(result)}")
print(f"Expected: 1")
```

Output:
```
Result: (2, 1)
Sum: 3
Expected: 1
```

## Why This Is A Bug

The function is supposed to generate zarr chunks that fit within a region of a given size. The fundamental invariant is that the chunks should sum to exactly the size parameter. When `chunk_size > size`, the function incorrectly creates a first chunk of size `chunk_size`, even though the total size is smaller. This violates the basic contract of the function and could lead to data corruption or out-of-bounds access in zarr backends.

## Fix

The issue is on line 146 where the first chunk size is calculated without considering that it might exceed the total size:

```diff
--- a/xarray/backends/chunks.py
+++ b/xarray/backends/chunks.py
@@ -143,7 +143,7 @@ def build_grid_chunks(

     region_start = region.start or 0
     # Generate the zarr chunks inside the region of this dim
-    chunks_on_region = [chunk_size - (region_start % chunk_size)]
+    chunks_on_region = [min(size, chunk_size - (region_start % chunk_size))]
     chunks_on_region.extend([chunk_size] * ((size - chunks_on_region[0]) // chunk_size))
     if (size - chunks_on_region[0]) % chunk_size != 0:
         chunks_on_region.append((size - chunks_on_region[0]) % chunk_size)
```
