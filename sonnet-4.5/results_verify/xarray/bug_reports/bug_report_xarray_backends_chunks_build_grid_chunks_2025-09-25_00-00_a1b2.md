# Bug Report: xarray.backends.chunks.build_grid_chunks Sum Invariant Violation

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`build_grid_chunks` violates the fundamental invariant that the sum of returned chunks must equal the input `size` parameter when `size < chunk_size`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.backends.chunks import build_grid_chunks

@given(
    size=st.integers(min_value=1, max_value=10000),
    chunk_size=st.integers(min_value=1, max_value=1000),
)
@settings(max_examples=1000)
def test_build_grid_chunks_sum_equals_size(size, chunk_size):
    chunks = build_grid_chunks(size=size, chunk_size=chunk_size, region=None)
    assert sum(chunks) == size, f"Sum of chunks {sum(chunks)} != size {size}"
```

**Failing input**: `size=1, chunk_size=2`

## Reproducing the Bug

```python
from xarray.backends.chunks import build_grid_chunks

result = build_grid_chunks(size=1, chunk_size=2, region=None)
print(f"Result: {result}")
print(f"Sum: {sum(result)}")
print(f"Expected: 1")

result2 = build_grid_chunks(size=5, chunk_size=10, region=None)
print(f"Result: {result2}")
print(f"Sum: {sum(result2)}")
print(f"Expected: 5")
```

**Output**:
```
Result: (2, 1)
Sum: 3
Expected: 1

Result: (10, 5)
Sum: 15
Expected: 5
```

## Why This Is A Bug

The function is documented and used in the codebase to partition a dimension of `size` elements into chunks. The fundamental invariant is that the sum of all chunk sizes must equal the total size. When `size < chunk_size`, the function incorrectly creates:
1. A first chunk of size `chunk_size` (larger than the entire dimension)
2. An additional chunk with the modulo remainder

This violates the core contract of the function and could lead to out-of-bounds memory access or data corruption when these chunks are used to partition actual data arrays.

## Fix

The bug is on line 146. The first chunk size should not exceed the total `size`:

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