# Bug Report: xarray.backends.chunks build_grid_chunks Sum Invariant Violation

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_grid_chunks` function violates its core invariant: the returned chunks must sum to exactly the input `size` parameter. When `size < chunk_size`, the function returns chunks that sum to more than `size`, breaking downstream code that relies on this invariant.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.backends.chunks import build_grid_chunks

@given(
    size=st.integers(min_value=1, max_value=1000),
    chunk_size=st.integers(min_value=1, max_value=100)
)
def test_build_grid_chunks_sum(size, chunk_size):
    chunks = build_grid_chunks(size, chunk_size)

    assert sum(chunks) == size, \
        f"Sum of chunks {sum(chunks)} != size {size}"

    assert all(c > 0 for c in chunks), \
        f"All chunks should be positive, got {chunks}"
```

**Failing input**: `size=1, chunk_size=2`

## Reproducing the Bug

```python
from xarray.backends.chunks import build_grid_chunks

result = build_grid_chunks(size=1, chunk_size=2)
print(f"Result: {result}")
print(f"Sum: {sum(result)}")
print(f"Expected: 1")

assert sum(result) == 1
```

**Output:**
```
Result: (2, 1)
Sum: 3
Expected: 1
AssertionError
```

## Why This Is A Bug

1. **Contract violation**: The function is called with `sum(var_chunks)` as the size parameter (see `grid_rechunk` at line 164), meaning the returned chunks must sum to exactly that size.

2. **Root cause**: When computing the first chunk as `chunk_size - (region_start % chunk_size)`, the code doesn't cap this value at the remaining size. For `size=1, chunk_size=2, region_start=0`:
   - First chunk = `2 - (0 % 2)` = `2` (but size is only 1!)
   - Remaining = `1 - 2` = `-1`
   - The negative arithmetic then adds a spurious chunk

3. **Real-world impact**: This affects users who:
   - Have small dimensions (e.g., single timestep) with larger chunk sizes
   - Use `align_chunks=True` when writing to Zarr format
   - Would result in incorrect chunk alignment and potential data corruption

## Fix

```diff
def build_grid_chunks(
    size: int,
    chunk_size: int,
    region: slice | None = None,
) -> tuple[int, ...]:
    if region is None:
        region = slice(0, size)

    region_start = region.start or 0
-    chunks_on_region = [chunk_size - (region_start % chunk_size)]
+    first_chunk = chunk_size - (region_start % chunk_size)
+    chunks_on_region = [min(first_chunk, size)]
    chunks_on_region.extend([chunk_size] * ((size - chunks_on_region[0]) // chunk_size))
    if (size - chunks_on_region[0]) % chunk_size != 0:
        chunks_on_region.append((size - chunks_on_region[0]) % chunk_size)
    return tuple(chunks_on_region)
```