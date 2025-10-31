# Bug Report: xarray.backends.chunks.build_grid_chunks Returns Incorrect Chunk Sum

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_grid_chunks` function returns chunks that sum to more than the specified `size` when `chunk_size > size`, violating its fundamental invariant that the returned chunks must sum to exactly `size`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.backends.chunks import build_grid_chunks

@given(
    size=st.integers(min_value=1, max_value=1000),
    chunk_size=st.integers(min_value=1, max_value=100)
)
def test_build_grid_chunks_preserves_size(size, chunk_size):
    result = build_grid_chunks(size, chunk_size)
    assert sum(result) == size
```

**Failing input**: `size=1, chunk_size=2`

## Reproducing the Bug

```python
from xarray.backends.chunks import build_grid_chunks

result = build_grid_chunks(size=1, chunk_size=2)
print(f"Result: {result}")
print(f"Sum: {sum(result)}, Expected: 1")

assert sum(result) == 1
```

Output:
```
Result: (2, 1)
Sum: 3, Expected: 1
AssertionError
```

## Why This Is A Bug

The function's contract requires that chunks sum to `size`, as evidenced by its usage in `grid_rechunk` (line 163-167) and the validation in `align_nd_chunks` (line 20-25):

```python
if sum(backend_chunks) != sum(var_chunks):
    raise ValueError(
        "The number of elements in the backend does not "
        "match the number of elements in the variable. "
        "This inconsistency should never occur at this stage."
    )
```

The bug occurs when `chunk_size > size`. The first chunk is incorrectly set to `chunk_size - (region_start % chunk_size)`, which equals `chunk_size` when `region_start % chunk_size == 0`. This doesn't account for cases where the total `size` is less than this value.

## Fix

```diff
diff --git a/xarray/backends/chunks.py b/xarray/backends/chunks.py
index 1234567..abcdefg 100644
--- a/xarray/backends/chunks.py
+++ b/xarray/backends/chunks.py
@@ -143,7 +143,8 @@ def build_grid_chunks(

     region_start = region.start or 0
     # Generate the zarr chunks inside the region of this dim
-    chunks_on_region = [chunk_size - (region_start % chunk_size)]
+    first_chunk = min(size, chunk_size - (region_start % chunk_size))
+    chunks_on_region = [first_chunk]
     chunks_on_region.extend([chunk_size] * ((size - chunks_on_region[0]) // chunk_size))
     if (size - chunks_on_region[0]) % chunk_size != 0:
         chunks_on_region.append((size - chunks_on_region[0]) % chunk_size)
```