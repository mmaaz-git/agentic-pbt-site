# Bug Report: xarray.backends.chunks.build_grid_chunks Incorrect Result for size=0

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_grid_chunks` function returns chunks that sum to `chunk_size` instead of `0` when `size=0`, violating the fundamental invariant that the sum of chunks should equal the size parameter.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from xarray.backends.chunks import build_grid_chunks

@given(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=1, max_value=100)
)
def test_build_grid_chunks_sum_equals_size(size, chunk_size):
    chunks = build_grid_chunks(size=size, chunk_size=chunk_size, region=None)
    assert sum(chunks) == size, \
        f"sum(chunks)={sum(chunks)} != size={size} for chunks={chunks}"
```

**Failing input**: `size=0, chunk_size=any positive integer`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.backends.chunks import build_grid_chunks

result = build_grid_chunks(size=0, chunk_size=5, region=None)
print(f"size=0, chunk_size=5")
print(f"Result: {result}")
print(f"Sum of chunks: {sum(result)}")
print(f"Expected: 0")
print(f"Bug: {sum(result)} != 0")
```

Output:
```
size=0, chunk_size=5
Result: (5,)
Sum of chunks: 5
Expected: 0
Bug: 5 != 0
```

## Why This Is A Bug

The function `build_grid_chunks` is supposed to divide a size into chunks. The fundamental invariant is that `sum(chunks) == size`. This is used by the caller `grid_rechunk` (line 164) which passes `sum(var_chunks)` as the size.

Tracing through the code for `size=0, chunk_size=5, region=None`:

1. Line 144: `region_start = 0`
2. Line 146: `chunks_on_region = [chunk_size - (region_start % chunk_size)] = [5]`
3. Line 147: `size - chunks_on_region[0] = 0 - 5 = -5`
4. `(-5) // 5 = -1`, so extend with `[5] * (-1) = []` (empty list)
5. `chunks_on_region = [5]`
6. Line 148: `(0 - 5) % 5 = 0`, so no append
7. Line 150: Returns `(5,)`
8. But `sum((5,)) = 5 ≠ 0`!

The bug occurs because the initial chunk size calculation (line 146) doesn't account for the case where `size < chunk_size - (region_start % chunk_size)`.

## Fix

```diff
--- a/xarray/backends/chunks.py
+++ b/xarray/backends/chunks.py
@@ -142,8 +142,13 @@ def build_grid_chunks(
         region = slice(0, size)

     region_start = region.start or 0
+
+    if size == 0:
+        return ()
+
     # Generate the zarr chunks inside the region of this dim
     chunks_on_region = [chunk_size - (region_start % chunk_size)]
+    chunks_on_region[0] = min(chunks_on_region[0], size)
     chunks_on_region.extend([chunk_size] * ((size - chunks_on_region[0]) // chunk_size))
     if (size - chunks_on_region[0]) % chunk_size != 0:
         chunks_on_region.append((size - chunks_on_region[0]) % chunk_size)
```

Alternative simpler fix:
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