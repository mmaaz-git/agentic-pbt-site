# Bug Report: xarray.backends.chunks.build_grid_chunks Incorrect Sum

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_grid_chunks` function returns chunks that sum to more than the specified `size` when `chunk_size > size`. This violates the fundamental invariant that the sum of chunks should equal the total size.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.backends.chunks import build_grid_chunks

@given(
    size=st.integers(min_value=1, max_value=1000),
    chunk_size=st.integers(min_value=1, max_value=100),
)
def test_build_grid_chunks_sum_equals_size(size, chunk_size):
    chunks = build_grid_chunks(size, chunk_size, region=None)
    assert sum(chunks) == size
```

**Failing input**: `size=1, chunk_size=2`

## Reproducing the Bug

```python
from xarray.backends.chunks import build_grid_chunks

result = build_grid_chunks(size=1, chunk_size=2, region=None)
print(f"Result: {result}")
print(f"Sum: {sum(result)}")
print(f"Expected sum: 1")

result = build_grid_chunks(size=5, chunk_size=10, region=None)
print(f"Result: {result}")
print(f"Sum: {sum(result)}")
print(f"Expected sum: 5")
```

Output:
```
Result: (2, 1)
Sum: 3
Expected sum: 1
Result: (10, 5)
Sum: 15
Expected sum: 5
```

## Why This Is A Bug

The function's purpose is to divide a region of `size` elements into chunks of approximately `chunk_size`. The sum of returned chunks must equal `size` to correctly represent the region. When `chunk_size > size`, the function should return a single chunk of `size`, but instead it returns `(chunk_size, size)`.

This bug occurs because line 146 unconditionally sets the first chunk to `chunk_size - (region_start % chunk_size)`, which equals `chunk_size` when `region_start=0`. This value can exceed the total `size`, violating the fundamental invariant.

The existing test suite in `test_backends_chunks.py` only tests cases where `size >= chunk_size`, missing this edge case entirely.

## Fix

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

This ensures the first chunk never exceeds the total `size`, fixing the bug while preserving correct behavior for all other cases.